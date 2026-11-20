# premortem: private-leaderboard pre-mortem of the candidate files and of the lead's standing assumptions (KEY = premortem)

Written 2026-11-15 (04:55Z onward) by the premortem agent. CPU only, stored predictions only, no training, no submission file. Work directory `nn_x/w12/stale/premortem/`. Conventions: ± = 1 SE paired per feed; val = 14,423 feeds; holdout = 30,000 in-period feeds (all M >= 50); lagged subset = 13,424 holdout feeds of days >= Sep 13; [M] measured by me, [D] read in a project file (path given), [I] inferred. FINAL = 0.5 z(FT branch) + 0.5 z(TO branch), FT branch on val = out-of-fold. pmix = `src/pmix.py` reweighting of val feeds to the public / private mix of (M bin x cold-share bin). stale-val = CPU fp32 re-prediction of the train-only members with private-like inputs (files of the witness and veto agents; CPU against CPU). SVF = "half-stale final" = 0.5 z(FT out-of-fold, fresh inputs) + 0.5 z(TO from the stale-val files) (my definition, information: no fine-tuned prediction exists under stale inputs). Rules R1-R7 and the dated blocks: `nn_x/w12/stale/premortem/PREDECLARED.txt`, each written before the read it judges.

Names. W8 / W16 = 8 / 16 wide students; W8b = the 8 newer ones (seeds 955-958); Pd = 4 popularity-light students d128; Pw = 4 popularity-light students d192 (fine-tunes landed 04:04-05:05Z); Px8 = Pd + Pw member-equal. "W16P40x8" = per branch 0.6 z(W16) + 0.4 z(Px8), and so on; suffix d / w / x8 = Pd / Pw / Px8. RULEM1040 = W16 + Px8 with popularity-light weight 0.1 on feeds with M < 50 and 0.4 on feeds with M >= 50; RULEM2040 = 0.2 / 0.4 (the scored weight kept on short feeds); RULEM2040w = the same with Pw. All new compositions exist in memory only.

## 0. Verdict

1. **Best file for private among everything scored: W8P20.** Forecast private delta against W16 +0.00071 ± 0.00040 (P(W8P20 better) 0.92 by the pre-declared rule; 0.85 as my own number after the lagged read and for shift nobody can model), against W8 +0.00066 ± 0.00033 (0.94). Val, pmix-private, half-stale val and public agree; label-free the two files differ on private days exactly as on val and public days; the gain is specific to the view (a same-view admixture adds 0.0000 on two disjoint 8-bases, the popularity-light one +0.0005 .. +0.0010) and public rewarded the view three times (+0.00052, +0.00057, +0.00070).
2. **The lead's pair W16 + W8P20 holds the right two files in the wrong order.** W16 is a hedge worth about +0.00002 in the expected maximum of the pair. The declared next file W16P20x8 is NOT forecast to beat W8P20 (-0.00034 ± 0.00049; P 0.27); the full-data form with 4 wide popularity-light members (held-out original W16P20w) is a tie with W8P20 (-0.00004 ± 0.00050) and +0.0007 above W16.
3. **Weight: 0.2 stands** for the scored form on every witness. For 16-base files with the wide popularity-light students the val-sample witnesses prefer 0.4 on long feeds (M >= 50): +0.00064 ± 0.00021 over constant 0.2 with the same members, nested +0.00045 .. +0.00065; **the lagged holdout that I scored for this says the opposite in 9 of 9 compositions (mean -0.00045 ± 0.00023)**: by my rules NOT defensible as a blind change. No staleness bonus on private: public-like against private-like lag moves the admixture by +0.00006 ± 0.00018.
4. **A number on record is weaker than quoted:** the lagged-holdout read of the admixture (+0.00024, "bias explanation holds") is the most favourable of nine compositions; their mean is +0.00003 ± 0.00023. The lagged holdout reads this family at zero where public reads +0.0005 .. +0.0007: it is not a veto, and it is not a confirmation either.
5. **Audit, 17 standing claims:** NOT SUPPORTED: "val under-reads the fine-tune half because 3 fold-models score a val row and 15 a test row" (the fold noise is 0.26% of the pool variance and worth about +0.0001 of the 0.0004 gap; the fine-tune half is worth more on colder feeds: pmix-private +0.0007 .. +0.0010, so it matters MORE on private and the fine-tune under stale inputs moves up the list). WEAKENED: "FT half adds 0 for students", "transfer coefficients are not identified" (slope of public on val 1.08 ± 0.49), "W16P20x8 is the W8P20 form", "private is staler so the gain is larger". Order not supported: pick 1 = W16. HOLD: w_ft 0.5, more FT seeds (public 0.0000 ± 0.0001), saturation at 16, narrow students, soft-target fine-tune (re-read at the private mix: -0.00044 ± 0.00046), holdout for non-staleness families.
6. **Pick today: W8P20 first, W16 as the hedge** (only scored files qualify). **The experiment that would most change it:** two public slots for the same 24 members at two weights, W16P20x8 (as declared) and RULEM2040 (0.2 on M < 50, 0.4 on M >= 50), CPU build. Their difference has a paired public SE of 0.00027; val predicts +0.0009 (pmix-public, in-sample; nested about +0.0005), the lagged holdout about -0.0003 and the in-period holdout about -0.0007 (their reads per long feed, -0.00043 and -0.00111, times the 61% long feeds). It decides the weight of every 16-base final and tests the evidence policy that the hz / xd / th arms are being judged by.

## 1. The witness table

Checks first [M] (`logs/t1_build.txt`): my NDCG equals `src/metrics.ndcg_per_feed` on a tied random score (max difference 0); my four branch pools of W8, W16, S16 reproduce `preds/ens_<tag>_*` (corr > 0.999999); G3 and C3 are read from the stored branches of the scored files (G3 has the one-seed fine-tune, C3 = `ens_Q2a8s3`); W8P20 - W8 reproduces the numbers on record on val (+0.00049 ± 0.00031), holdout (-0.00063 ± 0.00022) and CPU stale-val (+0.00088 ± 0.00036). The first build ran with 3 of 4 wide popularity-light fine-tunes (kept as `out/t1_table_3of4.md`); everything below is the re-run with 4 of 4.

### 1.1 Levels

| file | val FINAL | val FT branch | val TO branch | pmix-public | pmix-private | stale-val TO | SVF | holdout TO | PUBLIC |
|---|---|---|---|---|---|---|---|---|---|
| W8 | 0.40165 | 0.40087 | 0.40157 | 0.39860 | 0.40414 | 0.39982 | 0.40131 | 0.32456 | 0.394486 |
| W16 | 0.40158 | 0.40070 | 0.40144 | 0.39874 | 0.40429 | 0.39994 | 0.40084 | 0.32460 | 0.39448337 |
| S16 | 0.40173 | 0.40085 | 0.40172 | 0.39838 | 0.40376 | 0.39992 | 0.40123 | 0.32428 | 0.3940782 |
| W8P20 | 0.40214 | 0.40167 | 0.40212 | 0.39923 | 0.40493 | 0.40070 | 0.40195 | 0.32393 | 0.3950094 |
| W8TO | 0.40157 | 0.40157 | 0.40157 | 0.39794 | 0.40310 | 0.39982 | 0.39982 | 0.32456 | 0.394009 |
| C3 | 0.40101 | 0.40043 | 0.40103 | 0.39812 | 0.40378 | 0.39970 | 0.40040 | 0.32306 | 0.393454 |
| G3 | 0.40032 | 0.39984 | 0.40035 | 0.39705 | 0.40249 | 0.39941 | 0.39986 | 0.32379 | 0.39282 |
| W8b | 0.40139 | 0.40043 | 0.40122 | 0.39819 | 0.40360 | 0.39997 | 0.40080 | 0.32413 |  |
| W8P10 | 0.40195 | 0.40146 | 0.40200 | 0.39892 | 0.40456 | 0.40019 | 0.40177 | 0.32428 |  |
| W8P30 | 0.40177 | 0.40146 | 0.40177 | 0.39883 | 0.40454 | 0.40059 | 0.40212 | 0.32352 |  |
| W8P40 | 0.40195 | 0.40148 | 0.40170 | 0.39898 | 0.40480 | 0.40103 | 0.40201 | 0.32285 |  |
| W16P10d | 0.40178 | 0.40105 | 0.40166 | 0.39898 | 0.40454 | 0.40048 | 0.40125 | 0.32436 |  |
| W16P20d | 0.40172 | 0.40125 | 0.40152 | 0.39880 | 0.40433 | 0.40075 | 0.40157 | 0.32404 |  |
| W16P30d | 0.40202 | 0.40127 | 0.40156 | 0.39903 | 0.40467 | 0.40079 | 0.40204 | 0.32331 |  |
| W16P40d | 0.40207 | 0.40139 | 0.40144 | 0.39895 | 0.40460 | 0.40117 | 0.40211 | 0.32284 |  |
| W16P10x8 | 0.40181 | 0.40103 | 0.40180 | 0.39910 | 0.40472 | 0.40030 | 0.40131 | 0.32443 |  |
| W16P20x8 | 0.40189 | 0.40122 | 0.40173 | 0.39896 | 0.40451 | 0.40059 | 0.40159 | 0.32405 |  |
| W16P30x8 | 0.40220 | 0.40139 | 0.40175 | 0.39935 | 0.40507 | 0.40100 | 0.40187 | 0.32353 |  |
| W16P40x8 | 0.40244 | 0.40133 | 0.40202 | 0.39962 | 0.40543 | 0.40141 | 0.40218 | 0.32293 |  |
| W16P10w | 0.40170 | 0.40112 | 0.40180 | 0.39890 | 0.40453 | 0.40033 | 0.40143 | 0.32441 |  |
| W16P20w | 0.40214 | 0.40140 | 0.40190 | 0.39925 | 0.40490 | 0.40068 | 0.40184 | 0.32414 |  |
| W16P30w | 0.40236 | 0.40159 | 0.40220 | 0.39935 | 0.40494 | 0.40086 | 0.40186 | 0.32368 |  |
| W16P40w | 0.40236 | 0.40180 | 0.40237 | 0.39951 | 0.40527 | 0.40133 | 0.40202 | 0.32324 |  |
| W8P10x8 | 0.40204 | 0.40150 | 0.40175 | 0.39890 | 0.40445 | 0.40009 | 0.40184 | 0.32425 |  |
| W8P20x8 | 0.40224 | 0.40184 | 0.40200 | 0.39909 | 0.40470 | 0.40069 | 0.40181 | 0.32413 |  |
| W8P30x8 | 0.40216 | 0.40176 | 0.40201 | 0.39922 | 0.40496 | 0.40094 | 0.40170 | 0.32354 |  |
| W8P40x8 | 0.40218 | 0.40154 | 0.40219 | 0.39932 | 0.40515 | 0.40107 | 0.40185 | 0.32313 |  |
| W16TO | 0.40144 | 0.40144 | 0.40144 | 0.39821 | 0.40357 | 0.39994 | 0.39994 | 0.32460 |  |
| Pd | 0.40027 | 0.40033 | 0.39943 | 0.39730 | 0.40322 | 0.39908 | 0.40036 | 0.31715 |  |
| Pw | 0.40076 | 0.40099 | 0.40000 | 0.39833 | 0.40432 | 0.39997 | 0.40101 | 0.31772 |  |
| Px8 | 0.40093 | 0.40103 | 0.39987 | 0.39795 | 0.40372 | 0.39963 | 0.40094 | 0.31752 |  |

Feed cold share used by pmix (feed mean of the share of candidates with `it_hist_all_luo` = 0): val 0.290, public 0.371, private 0.406 [M]. (The 0.439 / 0.477 of the task text is a different definition; the ordering is the same.)

### 1.2 Deltas against W8 (the reference of the scored pairs)

| A - W8 | val FINAL | val M>=50 | val M<50 | val Tue | val Thu | pmix-public | pmix-private | stale-val TO | SVF | holdout TO (reported) | lagged holdout TO, subset / all 30K [D] | PUBLIC ± SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W16 | -0.00007 ± 0.00034 | -0.00024 | +0.00017 | +0.00013 | -0.00029 | +0.00014 ± 0.00042 | +0.00015 ± 0.00051 | +0.00012 ± 0.00034 | -0.00046 ± 0.00033 | +0.00004 ± 0.00019 |  | -0.000003 ± 0.00045 |
| S16 | +0.00008 ± 0.00034 | -0.00023 | +0.00052 | +0.00005 | +0.00011 | -0.00022 ± 0.00041 | -0.00038 ± 0.00049 | +0.00010 ± 0.00033 | -0.00008 ± 0.00034 | -0.00028 ± 0.00019 |  | -0.000408 ± 0.00044 |
| W8P20 | +0.00049 ± 0.00031 | +0.00080 | +0.00003 | +0.00100 | -0.00008 | +0.00063 ± 0.00039 | +0.00079 ± 0.00048 | +0.00088 ± 0.00036 | +0.00064 ± 0.00033 | -0.00063 ± 0.00022 | +0.00024 ± 0.00031 / +0.00020 ± 0.00021 | +0.000523 ± 0.00041 |
| W8TO | -0.00009 ± 0.00043 | -0.00002 | -0.00018 | -0.00005 | -0.00012 | -0.00066 ± 0.00054 | -0.00104 ± 0.00067 | +0.00000 ± 0.00000 | -0.00149 ± 0.00043 | +0.00000 ± 0.00000 |  | -0.000477 ± 0.00055 |
| C3 | -0.00064 ± 0.00055 | -0.00009 | -0.00144 | +0.00016 | -0.00151 | -0.00047 ± 0.00069 | -0.00036 ± 0.00085 | -0.00012 ± 0.00055 | -0.00091 ± 0.00056 | -0.00149 ± 0.00034 |  | -0.001032 ± 0.00071 |
| G3 | -0.00133 ± 0.00058 | -0.00086 | -0.00201 | -0.00059 | -0.00213 | -0.00155 ± 0.00073 | -0.00165 ± 0.00089 | -0.00041 ± 0.00057 | -0.00145 ± 0.00058 | -0.00076 ± 0.00033 |  | -0.001666 ± 0.00075 |
| W8b | -0.00026 ± 0.00048 | -0.00034 | -0.00014 | +0.00012 | -0.00068 | -0.00041 ± 0.00060 | -0.00054 ± 0.00074 | +0.00015 ± 0.00047 | -0.00051 ± 0.00047 | -0.00042 ± 0.00028 |  |  |
| W8P10 | +0.00029 ± 0.00021 | +0.00044 | +0.00009 | +0.00071 | -0.00016 | +0.00032 ± 0.00026 | +0.00042 ± 0.00032 | +0.00037 ± 0.00023 | +0.00046 ± 0.00023 | -0.00027 ± 0.00015 | +0.00013 ± 0.00021 / +0.00017 ± 0.00014 |  |
| W8P30 | +0.00012 ± 0.00040 | +0.00086 | -0.00094 | +0.00096 | -0.00080 | +0.00024 ± 0.00050 | +0.00040 ± 0.00062 | +0.00076 ± 0.00044 | +0.00081 ± 0.00041 | -0.00104 ± 0.00028 | -0.00015 ± 0.00038 / +0.00007 ± 0.00026 |  |
| W8P40 | +0.00030 ± 0.00047 | +0.00115 | -0.00093 | +0.00110 | -0.00058 | +0.00038 ± 0.00059 | +0.00066 ± 0.00074 | +0.00121 ± 0.00051 | +0.00070 ± 0.00047 | -0.00170 ± 0.00034 | -0.00053 ± 0.00045 / n/a |  |
| W16P10d | +0.00013 ± 0.00036 | +0.00004 | +0.00026 | +0.00039 | -0.00014 | +0.00038 ± 0.00044 | +0.00040 ± 0.00055 | +0.00066 ± 0.00036 | -0.00005 ± 0.00035 | -0.00019 ± 0.00021 |  |  |
| W16P20d | +0.00007 ± 0.00038 | +0.00052 | -0.00060 | +0.00055 | -0.00046 | +0.00020 ± 0.00047 | +0.00019 ± 0.00058 | +0.00093 ± 0.00041 | +0.00026 ± 0.00040 | -0.00052 ± 0.00025 |  |  |
| W16P30d | +0.00037 ± 0.00044 | +0.00118 | -0.00079 | +0.00099 | -0.00030 | +0.00043 ± 0.00053 | +0.00053 ± 0.00065 | +0.00096 ± 0.00046 | +0.00073 ± 0.00044 | -0.00124 ± 0.00030 |  |  |
| W16P40d | +0.00042 ± 0.00050 | +0.00125 | -0.00078 | +0.00124 | -0.00048 | +0.00035 ± 0.00061 | +0.00046 ± 0.00075 | +0.00134 ± 0.00052 | +0.00081 ± 0.00050 | -0.00172 ± 0.00035 |  |  |
| W16P10x8 | +0.00016 ± 0.00036 | +0.00009 | +0.00025 | +0.00051 | -0.00023 | +0.00050 ± 0.00045 | +0.00059 ± 0.00056 | +0.00048 ± 0.00036 | -0.00000 ± 0.00034 | -0.00013 ± 0.00021 |  |  |
| W16P20x8 | +0.00024 ± 0.00038 | +0.00072 | -0.00045 | +0.00080 | -0.00037 | +0.00036 ± 0.00047 | +0.00037 ± 0.00058 | +0.00077 ± 0.00040 | +0.00028 ± 0.00037 | -0.00051 ± 0.00025 |  |  |
| W16P30x8 | +0.00054 ± 0.00042 | +0.00137 | -0.00065 | +0.00149 | -0.00049 | +0.00075 ± 0.00051 | +0.00093 ± 0.00063 | +0.00118 ± 0.00045 | +0.00057 ± 0.00042 | -0.00103 ± 0.00030 |  |  |
| W16P40x8 | +0.00079 ± 0.00047 | +0.00179 | -0.00066 | +0.00159 | -0.00009 | +0.00103 ± 0.00059 | +0.00130 ± 0.00073 | +0.00159 ± 0.00051 | +0.00087 ± 0.00047 | -0.00162 ± 0.00034 |  |  |
| W16P10w | +0.00005 ± 0.00035 | +0.00004 | +0.00005 | +0.00036 | -0.00030 | +0.00030 ± 0.00045 | +0.00039 ± 0.00055 | +0.00051 ± 0.00036 | +0.00012 ± 0.00034 | -0.00014 ± 0.00021 |  |  |
| W16P20w | +0.00049 ± 0.00038 | +0.00054 | +0.00042 | +0.00104 | -0.00011 | +0.00065 ± 0.00048 | +0.00076 ± 0.00059 | +0.00086 ± 0.00040 | +0.00053 ± 0.00038 | -0.00042 ± 0.00025 |  |  |
| W16P30w | +0.00071 ± 0.00042 | +0.00142 | -0.00032 | +0.00128 | +0.00008 | +0.00076 ± 0.00052 | +0.00080 ± 0.00064 | +0.00104 ± 0.00046 | +0.00056 ± 0.00043 | -0.00088 ± 0.00029 |  |  |
| W16P40w | +0.00071 ± 0.00046 | +0.00173 | -0.00077 | +0.00145 | -0.00010 | +0.00091 ± 0.00058 | +0.00113 ± 0.00072 | +0.00151 ± 0.00052 | +0.00071 ± 0.00047 | -0.00132 ± 0.00034 |  |  |
| W8P10x8 | +0.00039 ± 0.00021 | +0.00060 | +0.00010 | +0.00061 | +0.00015 | +0.00031 ± 0.00025 | +0.00031 ± 0.00032 | +0.00027 ± 0.00024 | +0.00054 ± 0.00022 | -0.00030 ± 0.00015 |  |  |
| W8P20x8 | +0.00059 ± 0.00030 | +0.00092 | +0.00012 | +0.00103 | +0.00011 | +0.00049 ± 0.00038 | +0.00056 ± 0.00047 | +0.00086 ± 0.00034 | +0.00050 ± 0.00032 | -0.00043 ± 0.00022 |  |  |
| W8P30x8 | +0.00051 ± 0.00038 | +0.00127 | -0.00059 | +0.00134 | -0.00040 | +0.00062 ± 0.00049 | +0.00082 ± 0.00061 | +0.00112 ± 0.00043 | +0.00040 ± 0.00039 | -0.00101 ± 0.00028 |  |  |
| W8P40x8 | +0.00053 ± 0.00044 | +0.00155 | -0.00094 | +0.00122 | -0.00023 | +0.00072 ± 0.00056 | +0.00102 ± 0.00069 | +0.00125 ± 0.00049 | +0.00054 ± 0.00044 | -0.00143 ± 0.00033 |  |  |
| W16TO | -0.00022 ± 0.00045 | -0.00035 | -0.00002 | +0.00015 | -0.00061 | -0.00039 ± 0.00057 | -0.00057 ± 0.00070 | +0.00012 ± 0.00034 | -0.00136 ± 0.00046 | +0.00004 ± 0.00019 |  |  |

### 1.3 Deltas against W8P20 (the public-best file)

| A - W8P20 | val FINAL | val M>=50 | val M<50 | val Tue | val Thu | pmix-public | pmix-private | stale-val TO | SVF | holdout TO (reported) | lagged holdout TO, subset / all 30K [D] | PUBLIC ± SE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W8 | -0.00049 ± 0.00031 | -0.00080 | -0.00003 | -0.00100 | +0.00008 | -0.00063 ± 0.00039 | -0.00079 ± 0.00048 | -0.00088 ± 0.00036 | -0.00064 ± 0.00033 | +0.00063 ± 0.00022 |  | -0.000523 ± 0.00041 |
| W16 | -0.00056 ± 0.00038 | -0.00104 | +0.00013 | -0.00087 | -0.00021 | -0.00049 ± 0.00047 | -0.00064 ± 0.00057 | -0.00075 ± 0.00041 | -0.00110 ± 0.00038 | +0.00067 ± 0.00024 |  | -0.000526 ± 0.00050 |
| S16 | -0.00041 ± 0.00038 | -0.00103 | +0.00049 | -0.00095 | +0.00018 | -0.00085 ± 0.00047 | -0.00117 ± 0.00057 | -0.00078 ± 0.00041 | -0.00072 ± 0.00039 | +0.00035 ± 0.00024 |  | -0.000931 ± 0.00050 |
| W8P10 | -0.00019 ± 0.00022 | -0.00037 | +0.00006 | -0.00029 | -0.00008 | -0.00031 ± 0.00028 | -0.00037 ± 0.00035 | -0.00051 ± 0.00026 | -0.00018 ± 0.00024 | +0.00035 ± 0.00015 |  |  |
| W8P30 | -0.00036 ± 0.00023 | +0.00006 | -0.00098 | -0.00004 | -0.00072 | -0.00039 ± 0.00030 | -0.00039 ± 0.00037 | -0.00011 ± 0.00026 | +0.00017 ± 0.00023 | -0.00041 ± 0.00015 |  |  |
| W8P40 | -0.00019 ± 0.00034 | +0.00035 | -0.00096 | +0.00010 | -0.00051 | -0.00025 ± 0.00043 | -0.00013 ± 0.00053 | +0.00033 ± 0.00035 | +0.00006 ± 0.00032 | -0.00108 ± 0.00023 |  |  |
| W16P10d | -0.00035 ± 0.00034 | -0.00076 | +0.00023 | -0.00061 | -0.00007 | -0.00025 ± 0.00041 | -0.00039 ± 0.00051 | -0.00022 ± 0.00035 | -0.00069 ± 0.00034 | +0.00043 ± 0.00019 |  |  |
| W16P20d | -0.00042 ± 0.00030 | -0.00028 | -0.00063 | -0.00045 | -0.00038 | -0.00043 ± 0.00036 | -0.00060 ± 0.00044 | +0.00005 ± 0.00031 | -0.00038 ± 0.00031 | +0.00011 ± 0.00016 |  |  |
| W16P30d | -0.00012 ± 0.00032 | +0.00037 | -0.00083 | -0.00001 | -0.00022 | -0.00019 ± 0.00039 | -0.00026 ± 0.00047 | +0.00009 ± 0.00033 | +0.00009 ± 0.00031 | -0.00062 ± 0.00020 |  |  |
| W16P40d | -0.00007 ± 0.00038 | +0.00045 | -0.00082 | +0.00024 | -0.00041 | -0.00028 ± 0.00047 | -0.00033 ± 0.00057 | +0.00047 ± 0.00038 | +0.00017 ± 0.00037 | -0.00109 ± 0.00024 |  |  |
| W16P10x8 | -0.00033 ± 0.00034 | -0.00071 | +0.00022 | -0.00049 | -0.00016 | -0.00013 ± 0.00041 | -0.00020 ± 0.00051 | -0.00040 ± 0.00035 | -0.00064 ± 0.00034 | +0.00050 ± 0.00020 |  |  |
| W16P20x8 | -0.00025 ± 0.00032 | -0.00008 | -0.00048 | -0.00020 | -0.00030 | -0.00027 ± 0.00039 | -0.00041 ± 0.00048 | -0.00011 ± 0.00031 | -0.00036 ± 0.00031 | +0.00012 ± 0.00017 |  |  |
| W16P30x8 | +0.00006 ± 0.00033 | +0.00057 | -0.00068 | +0.00049 | -0.00041 | +0.00012 ± 0.00040 | +0.00014 ± 0.00048 | +0.00030 ± 0.00031 | -0.00007 ± 0.00032 | -0.00040 ± 0.00020 |  |  |
| W16P40x8 | +0.00030 ± 0.00037 | +0.00099 | -0.00069 | +0.00059 | -0.00001 | +0.00040 ± 0.00047 | +0.00051 ± 0.00057 | +0.00071 ± 0.00037 | +0.00023 ± 0.00036 | -0.00100 ± 0.00024 |  |  |
| W16P20w | +0.00000 ± 0.00034 | -0.00026 | +0.00039 | +0.00004 | -0.00004 | +0.00002 ± 0.00041 | -0.00002 ± 0.00050 | -0.00002 ± 0.00033 | -0.00011 ± 0.00034 | +0.00021 ± 0.00018 |  |  |
| W16P30w | +0.00022 ± 0.00036 | +0.00062 | -0.00035 | +0.00028 | +0.00016 | +0.00013 ± 0.00045 | +0.00002 ± 0.00054 | +0.00016 ± 0.00035 | -0.00008 ± 0.00035 | -0.00025 ± 0.00021 |  |  |
| W16P40w | +0.00022 ± 0.00040 | +0.00093 | -0.00081 | +0.00045 | -0.00003 | +0.00028 ± 0.00050 | +0.00034 ± 0.00061 | +0.00063 ± 0.00041 | +0.00007 ± 0.00039 | -0.00069 ± 0.00025 |  |  |
| W8P20x8 | +0.00010 ± 0.00019 | +0.00012 | +0.00009 | +0.00003 | +0.00019 | -0.00014 ± 0.00024 | -0.00023 ± 0.00029 | -0.00001 ± 0.00018 | -0.00014 ± 0.00019 | +0.00020 ± 0.00010 |  |  |
| W8P30x8 | +0.00002 ± 0.00026 | +0.00047 | -0.00062 | +0.00034 | -0.00032 | -0.00001 ± 0.00033 | +0.00003 ± 0.00041 | +0.00025 ± 0.00026 | -0.00024 ± 0.00024 | -0.00039 ± 0.00016 |  |  |
| W8P40x8 | +0.00004 ± 0.00032 | +0.00075 | -0.00098 | +0.00022 | -0.00015 | +0.00009 ± 0.00041 | +0.00023 ± 0.00050 | +0.00037 ± 0.00034 | -0.00010 ± 0.00031 | -0.00080 ± 0.00022 |  |  |
| C3 | -0.00113 ± 0.00051 | -0.00089 | -0.00147 | -0.00084 | -0.00143 | -0.00110 ± 0.00064 | -0.00114 ± 0.00079 | -0.00100 ± 0.00051 | -0.00155 ± 0.00051 | -0.00087 ± 0.00029 |  | -0.001555 ± 0.00066 |

Deltas against W16: `nn_x/w12/stale/premortem/out/t1_table.md`, table T1c.

### 1.4 Per branch

| A - ref | FT branch val | FT branch pmix-private | TO branch val | TO branch pmix-private | TO stale-val | TO stale-val M>=50 | TO stale-val M<50 |
|---|---|---|---|---|---|---|---|
| W16 - W8 | -0.00017 ± 0.00033 | -0.00031 ± 0.00050 | -0.00013 ± 0.00031 | +0.00048 ± 0.00049 | +0.00012 ± 0.00034 | -0.00027 ± 0.00034 | +0.00070 ± 0.00067 |
| S16 - W8 | -0.00002 ± 0.00036 | +0.00003 ± 0.00054 | +0.00015 ± 0.00033 | +0.00051 ± 0.00051 | +0.00010 ± 0.00033 | +0.00006 ± 0.00035 | +0.00016 ± 0.00064 |
| W8P20 - W8 | +0.00080 ± 0.00033 | +0.00071 ± 0.00052 | +0.00055 ± 0.00035 | +0.00124 ± 0.00052 | +0.00088 ± 0.00036 | +0.00172 ± 0.00039 | -0.00034 ± 0.00066 |
| C3 - G3 | +0.00059 ± 0.00039 | +0.00073 ± 0.00058 | +0.00068 ± 0.00045 | +0.00117 ± 0.00070 | +0.00029 ± 0.00046 | +0.00175 ± 0.00050 | -0.00182 ± 0.00087 |
| W8 - C3 | +0.00045 ± 0.00056 | +0.00046 ± 0.00085 | +0.00054 ± 0.00056 | +0.00004 ± 0.00084 | +0.00012 ± 0.00055 | -0.00053 ± 0.00060 | +0.00107 ± 0.00105 |
| W16P20d - W16 | +0.00055 ± 0.00030 | +0.00078 ± 0.00048 | +0.00008 ± 0.00032 | +0.00026 ± 0.00049 | +0.00081 ± 0.00035 | +0.00175 ± 0.00037 | -0.00056 ± 0.00065 |
| W16P20x8 - W16 | +0.00051 ± 0.00029 | +0.00050 ± 0.00047 | +0.00030 ± 0.00031 | +0.00051 ± 0.00048 | +0.00065 ± 0.00033 | +0.00171 ± 0.00035 | -0.00090 ± 0.00063 |
| W16P30x8 - W16 | +0.00069 ± 0.00035 | +0.00079 ± 0.00056 | +0.00032 ± 0.00040 | +0.00047 ± 0.00060 | +0.00106 ± 0.00041 | +0.00205 ± 0.00044 | -0.00038 ± 0.00078 |
| W16P40x8 - W16 | +0.00063 ± 0.00041 | +0.00103 ± 0.00064 | +0.00059 ± 0.00047 | +0.00106 ± 0.00072 | +0.00147 ± 0.00048 | +0.00243 ± 0.00052 | +0.00007 ± 0.00092 |
| W16P20w - W16 | +0.00069 ± 0.00030 | +0.00058 ± 0.00049 | +0.00047 ± 0.00031 | +0.00050 ± 0.00048 | +0.00074 ± 0.00033 | +0.00183 ± 0.00035 | -0.00085 ± 0.00062 |
| W16P40w - W16 | +0.00110 ± 0.00042 | +0.00167 ± 0.00067 | +0.00093 ± 0.00047 | +0.00118 ± 0.00071 | +0.00138 ± 0.00049 | +0.00238 ± 0.00053 | -0.00006 ± 0.00093 |
| W8P30 - W8 | +0.00059 ± 0.00039 | +0.00073 ± 0.00061 | +0.00021 ± 0.00042 | +0.00101 ± 0.00062 | +0.00076 ± 0.00044 | +0.00196 ± 0.00047 | -0.00097 ± 0.00083 |
| W8P40 - W8 | +0.00061 ± 0.00045 | +0.00081 ± 0.00071 | +0.00013 ± 0.00051 | +0.00145 ± 0.00076 | +0.00121 ± 0.00051 | +0.00197 ± 0.00055 | +0.00009 ± 0.00094 |
| W16P20x8 - W8P20 | -0.00045 ± 0.00030 | -0.00051 ± 0.00048 | -0.00038 ± 0.00032 | -0.00025 ± 0.00046 | -0.00011 ± 0.00031 | -0.00027 ± 0.00034 | +0.00014 ± 0.00059 |
| W16P20d - W8P20 | -0.00042 ± 0.00030 | -0.00023 ± 0.00047 | -0.00060 ± 0.00031 | -0.00050 ± 0.00045 | +0.00005 ± 0.00031 | -0.00024 ± 0.00033 | +0.00048 ± 0.00059 |
| W16 - W8P20 | -0.00097 ± 0.00037 | -0.00101 ± 0.00057 | -0.00068 ± 0.00038 | -0.00076 ± 0.00058 | -0.00075 ± 0.00041 | -0.00199 ± 0.00043 | +0.00103 ± 0.00079 |

### 1.5 Sanity comparison with the scored pairs (rule R7, nothing fitted)

| pair | PUBLIC ± SE | val FINAL | pmix-public | pmix-private | SVF | TO val | TO stale-val | holdout TO |
|---|---|---|---|---|---|---|---|---|
| W16 - W8 | -0.000003 ± 0.00045 | -0.00007 ± 0.00034 | +0.00014 ± 0.00042 | +0.00015 ± 0.00051 | -0.00046 ± 0.00033 | -0.00013 ± 0.00031 | +0.00012 ± 0.00034 | +0.00004 ± 0.00019 |
| W8P20 - W8 | +0.000523 ± 0.00041 | +0.00049 ± 0.00031 | +0.00063 ± 0.00039 | +0.00079 ± 0.00048 | +0.00064 ± 0.00033 | +0.00055 ± 0.00035 | +0.00088 ± 0.00036 | -0.00063 ± 0.00022 |
| W8TO - W8 | -0.000477 ± 0.00055 | -0.00009 ± 0.00043 | -0.00066 ± 0.00054 | -0.00104 ± 0.00067 | -0.00149 ± 0.00043 | +0.00000 ± 0.00000 | +0.00000 ± 0.00000 | +0.00000 ± 0.00000 |
| S16 - W8 | -0.000408 ± 0.00044 | +0.00008 ± 0.00034 | -0.00022 ± 0.00041 | -0.00038 ± 0.00049 | -0.00008 ± 0.00034 | +0.00015 ± 0.00033 | +0.00010 ± 0.00033 | -0.00028 ± 0.00019 |
| C3 - G3 | +0.000634 ± 0.00052 | +0.00069 ± 0.00040 | +0.00108 ± 0.00050 | +0.00130 ± 0.00062 | +0.00054 ± 0.00041 | +0.00068 ± 0.00045 | +0.00029 ± 0.00046 | -0.00073 ± 0.00029 |
| W8 - C3 | +0.001032 ± 0.00071 | +0.00064 ± 0.00055 | +0.00047 ± 0.00069 | +0.00036 ± 0.00085 | +0.00091 ± 0.00056 | +0.00054 ± 0.00056 | +0.00012 ± 0.00055 | +0.00149 ± 0.00034 |
| G3 - W8 | -0.001666 ± 0.00075 | -0.00133 ± 0.00058 | -0.00155 ± 0.00073 | -0.00165 ± 0.00089 | -0.00145 ± 0.00058 | -0.00121 ± 0.00056 | -0.00041 ± 0.00057 | -0.00076 ± 0.00033 |

| witness column | sign right on pairs with abs(public) >= 1 public SE | sign right, all 6 independent pairs | X2 (6 pairs) | slope through the origin of public on the column (6 pairs) |
|---|---|---|---|---|
| val FINAL | 3 / 3 | 5 / 6 | 0.22 | +1.19 |
| pmix-public FINAL | 3 / 3 | 5 / 6 | 0.16 | +0.84 |
| pmix-private FINAL | 3 / 3 | 5 / 6 | 0.28 | +0.61 |
| SVF | 3 / 3 | 6 / 6 | 0.53 | +0.60 |
| TO val | 3 / 3 | 4 / 6 | 0.36 | +1.11 |
| TO stale-val | 3 / 3 | 3 / 6 | 0.56 | +0.81 |
| holdout TO | 1 / 3 | 2 / 6 | 2.10 | +0.27 |
(W8TO - W8 has no train-only difference on the TO columns by construction: the TO columns of that pair are 0 and count as wrong sign.)

Reading [M]. (1) At FINAL level plain val, pmix-public and pmix-private all have the public sign on the three pairs that are at least 1 public SE away from zero, and on 5 of 6 overall (the misses are S16 - W8 for val and W16 - W8 for pmix; the second has a public delta of 0.000003). X2: pmix-public 0.16, val 0.22, pmix-private 0.28, SVF 0.53, in-period holdout 2.10. (2) The holdout is wrong only on the popularity-light pairs; on the three other pairs (W16 - W8, S16 - W8, W8 - C3) it has the public sign and size, so the two-witness rule for non-staleness families is not contradicted by these readings. (3) SVF over-reads the value of the fine-tune half (W8TO - W8: -0.00149 against public -0.00048) because only one branch is made stale; it is an upper bound for "the fine-tune is immune to staleness". (4) Slope of public on the column over the 6 pairs: unweighted 1.19 (val FINAL), 0.84 (pmix-public), 0.61 (pmix-private) in the table above; weighted by the public SEs, with its SE: 1.08 ± 0.49, 0.78 ± 0.34, 0.59 ± 0.26 (table T5c in section 4.3).

## 2. Private forecast

### 2.1 The forecast (rule R1) for the short list fixed in advance: W8P20, W16, W8, W16P20d, W16P20x8

Forecast = inverse-variance mean of (a) the centre of three views of the val sample (plain val, pmix-private, SVF; sampling SE = the largest of the three, plus half their range as model spread) and (b) the public delta shifted by (pmix-private - pmix-public) where a public delta exists. "Realised" adds the sampling SE of the private part itself (20,033 feeds).

| A - B | V1 val FINAL | V2 pmix-private | V3 SVF | centre c_val ± sd (sampling, spread) | PUBLIC ± SE | public shifted to private ± sd | **FORECAST private delta ± sd** | 90% interval of the REALISED private delta | P(A > B) expected / realised |
|---|---|---|---|---|---|---|---|---|---|
| W16 - W8P20 | -0.00056 ± 0.00038 | -0.00064 ± 0.00057 | -0.00110 ± 0.00038 | -0.00077 ± 0.00063 (0.00057, 0.00027) | -0.000526 ± 0.00050 | -0.00068 ± 0.00052 | **-0.00071 ± 0.00040** | [-0.00156, +0.00013] | 0.04 / 0.08 |
| W8 - W8P20 | -0.00049 ± 0.00031 | -0.00079 ± 0.00048 | -0.00064 ± 0.00033 | -0.00064 ± 0.00051 (0.00048, 0.00015) | -0.000523 ± 0.00041 | -0.00068 ± 0.00044 | **-0.00066 ± 0.00033** | [-0.00137, +0.00004] | 0.02 / 0.06 |
| W16P20d - W8P20 | -0.00042 ± 0.00030 | -0.00060 ± 0.00044 | -0.00038 ± 0.00031 | -0.00047 ± 0.00045 (0.00044, 0.00011) | none |  | **-0.00047 ± 0.00045** | [-0.00132, +0.00039] | 0.15 / 0.18 |
| W16P20x8 - W8P20 | -0.00025 ± 0.00032 | -0.00041 ± 0.00048 | -0.00036 ± 0.00031 | -0.00034 ± 0.00049 (0.00048, 0.00008) | none |  | **-0.00034 ± 0.00049** | [-0.00126, +0.00058] | 0.24 / 0.27 |

| A - B | V1 val FINAL | V2 pmix-private | V3 SVF | centre c_val ± sd (sampling, spread) | PUBLIC ± SE | public shifted to private ± sd | **FORECAST private delta ± sd** | 90% interval of the REALISED private delta | P(A > B) expected / realised |
|---|---|---|---|---|---|---|---|---|---|
| W8P20 - W16 | +0.00056 ± 0.00038 | +0.00064 ± 0.00057 | +0.00110 ± 0.00038 | +0.00077 ± 0.00063 (0.00057, 0.00027) | +0.000526 ± 0.00050 | +0.00068 ± 0.00052 | **+0.00071 ± 0.00040** | [-0.00013, +0.00156] | 0.96 / 0.92 |
| W8 - W16 | +0.00007 ± 0.00034 | -0.00015 ± 0.00051 | +0.00046 ± 0.00033 | +0.00013 ± 0.00060 (0.00051, 0.00031) | +0.000003 ± 0.00045 | -0.00000 ± 0.00045 | **+0.00004 ± 0.00036** | [-0.00071, +0.00080] | 0.55 / 0.54 |
| W16P20d - W16 | +0.00014 ± 0.00031 | +0.00004 ± 0.00048 | +0.00072 ± 0.00030 | +0.00030 ± 0.00058 (0.00048, 0.00034) | none |  | **+0.00030 ± 0.00058** | [-0.00076, +0.00136] | 0.70 / 0.68 |
| W16P20x8 - W16 | +0.00031 ± 0.00031 | +0.00023 ± 0.00046 | +0.00075 ± 0.00029 | +0.00043 ± 0.00053 (0.00046, 0.00026) | none |  | **+0.00043 ± 0.00053** | [-0.00054, +0.00140] | 0.79 / 0.77 |

**The file I expect to be best on private among the scored files and the short list: W8P20.** P(W8P20 beats W16 on private) = 0.92; P(beats W8) = 0.94; P(beats the lead's declared W16P20x8) = 0.73; P(beats W16P20d) = 0.82 [by rule R1; I]. The expected margin over W16 is +0.0007 (90% interval of the realised delta -0.0001 .. +0.0016). W8 against W16 is a coin flip (+0.00004 ± 0.00036). Rule R1 does not use the lagged holdout; after my lagged read (section 3.4: under the private-like lag the train-only pool of W16 equals the train-only half of W8P20, +0.00000 ± 0.00034, and the admixture reads +0.00003 ± 0.00023 over nine compositions) and for shift that no witness covers, my own number for P(W8P20 beats W16 on private) is 0.85.

Winner's curse of the short list [M]: picking the list member with the largest centre on one user half and scoring it on the other (200 splits) picks W8P20 in 83% of the half-samples and reads -0.00013 against "always W8P20": the list contains nothing that a selection on val would prefer to W8P20.

### 2.2 The four sources, one by one, for the popularity-light contrast W8P20 - W8

| source | reading | what it says about private |
|---|---|---|
| (a) pmix-private, FINAL | +0.00079 ± 0.00048 (pmix-public +0.00063 ± 0.00039; public +0.00052 ± 0.00041) | +0.0002 more than public; but pmix-private over-reads the six public pairs by a factor 1.7 (slope 0.59 ± 0.26), so I do not add the +0.0002 |
| (b) stale-val, TO branch | +0.00088 ± 0.00036; minus the fresh TO delta: +0.00032 ± 0.00041; SVF +0.00064 ± 0.00033; by day under stale inputs: Tue +0.00097, Thu +0.00078 | private-like inputs do not reduce the gain; they raise it by about +0.0003 at TO level (0.8 SE) |
| (c) staleness gradient | in-period holdout (1.75 d) -0.00063 ± 0.00022, val (5.69 d) +0.00049 ± 0.00031, public (9.34 d) +0.00052 ± 0.00041 [D]; lagged holdout: private-like minus public-like lag +0.00006 ± 0.00018 [D poplag_verify 1.3]; within val: Tue +0.00100 ± 0.00045, Thu -0.00008 ± 0.00043 (fresh inputs), no monotone trend over the four cold-share bins (table below) | a straight line through the three points gives +0.0010 at 11.2 d, but the curve is flat between val and public (+0.00001 per day) and the direct manipulation of the lag says +0.00006: the gradient implies private = public, about +0.0005, not more. The holdout-to-val jump is "in period against out of period", not a dose of staleness that keeps growing |
| (d) label-free behaviour by day | same per-feed correlation (0.99940 / 0.99939) and same top-1 agreement (93.4% / 93.4%) on public and private days; the cold-minus-warm offset of the admixture is +0.0064 on public days and +0.0055 on private days (val +0.0055 / +0.0057) | the two files differ on private days exactly as they do on val and public days: no drift, the public reading transfers; if anything the admixture acts 14% less on private than on public |

Forecast for W8P20 - W8 on private: **+0.00066 ± 0.00033** (expected), realised 90% interval -0.0000 .. +0.0014. Two further scored contrasts of the same view on older pools agree [M, table T5a]: C1 - G3 public +0.00057, C3 - C2 public +0.00070 (pmix-public +0.00090 / +0.00085).

The gradient table (val slices; shares of public / private feeds per slice):

| val slice | feeds | share of public / private feeds | W8P20 - W8 FINAL | W8P20 - W8 stale TO | W16P20x8 - W16 FINAL | W16P20x8 - W16 stale TO | W16P40x8 - W16 FINAL | W16P40x8 - W16 stale TO |
|---|---|---|---|---|---|---|---|---|
| all val | 14423 | 1.000 / 1.000 | +0.00049 ± 0.00031 | +0.00088 ± 0.00036 | +0.00031 ± 0.00031 | +0.00065 ± 0.00033 | +0.00086 ± 0.00044 | +0.00147 ± 0.00048 |
| val Tue 09-22 (2 d after the dense cutoff) | 7542 |  | +0.00100 ± 0.00045 | +0.00097 ± 0.00050 | +0.00068 ± 0.00045 | +0.00058 ± 0.00048 | +0.00147 ± 0.00064 | +0.00138 ± 0.00069 |
| val Thu 09-24 (4 d) | 6878 |  | -0.00008 ± 0.00043 | +0.00078 ± 0.00050 | -0.00008 ± 0.00042 | +0.00073 ± 0.00045 | +0.00020 ± 0.00061 | +0.00156 ± 0.00068 |
| cold share < 0.15 | 1370 | 0.018 / 0.011 | -0.00035 ± 0.00132 | +0.00067 ± 0.00134 | -0.00055 ± 0.00126 | +0.00125 ± 0.00106 | +0.00069 ± 0.00167 | +0.00073 ± 0.00164 |
| 0.15-0.30 | 6763 | 0.240 / 0.147 | +0.00053 ± 0.00043 | +0.00123 ± 0.00052 | +0.00083 ± 0.00042 | +0.00078 ± 0.00049 | +0.00170 ± 0.00060 | +0.00168 ± 0.00070 |
| 0.30-0.45 | 5077 | 0.527 / 0.514 | +0.00046 ± 0.00049 | +0.00057 ± 0.00056 | +0.00000 ± 0.00051 | +0.00020 ± 0.00054 | -0.00041 ± 0.00076 | +0.00088 ± 0.00081 |
| >= 0.45 | 1213 | 0.215 / 0.327 | +0.00128 ± 0.00134 | +0.00040 ± 0.00128 | -0.00029 ± 0.00114 | +0.00112 ± 0.00111 | +0.00168 ± 0.00173 | +0.00359 ± 0.00174 |
| M < 50 | 5893 | 0.387 / 0.392 | +0.00003 ± 0.00058 | -0.00034 ± 0.00066 | -0.00062 ± 0.00057 | -0.00090 ± 0.00063 | -0.00083 ± 0.00080 | +0.00007 ± 0.00092 |
| M >= 50 | 8530 | 0.613 / 0.608 | +0.00080 ± 0.00035 | +0.00172 ± 0.00039 | +0.00096 ± 0.00034 | +0.00171 ± 0.00035 | +0.00203 ± 0.00051 | +0.00243 ± 0.00052 |
| M >= 50 and cold >= 0.30 | 3641 | 0.472 / 0.531 | +0.00062 ± 0.00054 | +0.00147 ± 0.00060 | +0.00061 ± 0.00057 | +0.00160 ± 0.00050 | +0.00173 ± 0.00086 | +0.00279 ± 0.00080 |
| M >= 50 and cold < 0.30 | 4889 | 0.141 / 0.077 | +0.00093 ± 0.00046 | +0.00190 ± 0.00051 | +0.00121 ± 0.00042 | +0.00180 ± 0.00048 | +0.00225 ± 0.00061 | +0.00217 ± 0.00068 |
| M < 50 and cold >= 0.30 | 2649 | 0.270 / 0.310 | +0.00061 ± 0.00084 | -0.00074 ± 0.00091 | -0.00096 ± 0.00079 | -0.00130 ± 0.00093 | -0.00239 ± 0.00116 | -0.00050 ± 0.00135 |
| M < 50 and cold < 0.30 | 3244 | 0.117 / 0.082 | -0.00044 ± 0.00079 | -0.00001 ± 0.00095 | -0.00034 ± 0.00080 | -0.00057 ± 0.00086 | +0.00044 ± 0.00110 | +0.00053 ± 0.00125 |

Label-free day tables (three of ten pairs; the rest in `out/t2_forecast.md` T2h).

W8P20 against W8:

| group | feeds | mean cold share | per-feed Pearson r of z | same top-1 item | cold-minus-warm offset, A - B (z units) |
|---|---|---|---|---|---|
| val Tue 09-22 | 7542 | 0.258 | 0.99941 ± 0.00000 | 0.9390 ± 0.0028 | +0.0055 ± 0.0002 |
| val Thu 09-24 | 6878 | 0.324 | 0.99941 ± 0.00001 | 0.9365 ± 0.0029 | +0.0057 ± 0.0002 |
| Sat 09-26 (public) | 6050 | 0.367 | 0.99940 ± 0.00001 | 0.9337 ± 0.0032 | +0.0063 ± 0.0002 |
| Sun 09-27 am (public) | 2512 | 0.378 | 0.99940 ± 0.00001 | 0.9331 ± 0.0050 | +0.0065 ± 0.0003 |
| Sun 09-27 pm (private) | 4855 | 0.392 | 0.99940 ± 0.00001 | 0.9374 ± 0.0035 | +0.0054 ± 0.0002 |
| Mon 09-28 (private) | 7578 | 0.403 | 0.99939 ± 0.00001 | 0.9348 ± 0.0028 | +0.0056 ± 0.0002 |
| Tue 09-29 (private) | 7600 | 0.418 | 0.99940 ± 0.00001 | 0.9317 ± 0.0029 | +0.0054 ± 0.0002 |
| PUBLIC | 8562 | 0.371 | 0.99940 ± 0.00001 | 0.9335 ± 0.0027 | +0.0064 ± 0.0002 |
| PRIVATE | 20033 | 0.406 | 0.99939 ± 0.00000 | 0.9343 ± 0.0018 | +0.0055 ± 0.0001 |

W16P40x8 against W8P20:

| group | feeds | mean cold share | per-feed Pearson r of z | same top-1 item | cold-minus-warm offset, A - B (z units) |
|---|---|---|---|---|---|
| val Tue 09-22 | 7542 | 0.258 | 0.99884 ± 0.00001 | 0.9175 ± 0.0032 | +0.0069 ± 0.0003 |
| val Thu 09-24 | 6878 | 0.324 | 0.99884 ± 0.00001 | 0.9113 ± 0.0034 | +0.0073 ± 0.0003 |
| Sat 09-26 (public) | 6050 | 0.367 | 0.99881 ± 0.00001 | 0.9132 ± 0.0036 | +0.0080 ± 0.0003 |
| Sun 09-27 am (public) | 2512 | 0.378 | 0.99880 ± 0.00002 | 0.9112 ± 0.0057 | +0.0089 ± 0.0004 |
| Sun 09-27 pm (private) | 4855 | 0.392 | 0.99882 ± 0.00001 | 0.9125 ± 0.0041 | +0.0071 ± 0.0003 |
| Mon 09-28 (private) | 7578 | 0.403 | 0.99879 ± 0.00001 | 0.9165 ± 0.0032 | +0.0074 ± 0.0003 |
| Tue 09-29 (private) | 7600 | 0.418 | 0.99882 ± 0.00001 | 0.9145 ± 0.0032 | +0.0074 ± 0.0003 |
| PUBLIC | 8562 | 0.371 | 0.99881 ± 0.00001 | 0.9126 ± 0.0031 | +0.0083 ± 0.0003 |
| PRIVATE | 20033 | 0.406 | 0.99881 ± 0.00001 | 0.9147 ± 0.0020 | +0.0073 ± 0.0002 |

W8TO against W8 (= what the fine-tune half does): the only pair with a drift. The cold promotion of the fine-tune half grows with the calendar day (val Tue -0.0094, val Thu -0.0136, public -0.0169, private -0.0195, last private day -0.0210, in z units; on record in ft_redesign finding 7) and is larger on private than on anything val covers.

| group | feeds | mean cold share | per-feed Pearson r of z | same top-1 item | cold-minus-warm offset, A - B (z units) |
|---|---|---|---|---|---|
| val Tue 09-22 | 7542 | 0.258 | 0.99779 ± 0.00002 | 0.8982 ± 0.0035 | -0.0094 ± 0.0004 |
| val Thu 09-24 | 6878 | 0.324 | 0.99780 ± 0.00002 | 0.8979 ± 0.0037 | -0.0136 ± 0.0004 |
| Sat 09-26 (public) | 6050 | 0.367 | 0.99804 ± 0.00002 | 0.9068 ± 0.0037 | -0.0162 ± 0.0004 |
| Sun 09-27 am (public) | 2512 | 0.378 | 0.99801 ± 0.00002 | 0.9092 ± 0.0057 | -0.0184 ± 0.0006 |
| Sun 09-27 pm (private) | 4855 | 0.392 | 0.99806 ± 0.00002 | 0.9046 ± 0.0042 | -0.0174 ± 0.0004 |
| Mon 09-28 (private) | 7578 | 0.403 | 0.99805 ± 0.00002 | 0.9082 ± 0.0033 | -0.0193 ± 0.0003 |
| Tue 09-29 (private) | 7600 | 0.418 | 0.99801 ± 0.00002 | 0.9049 ± 0.0034 | -0.0210 ± 0.0003 |
| PUBLIC | 8562 | 0.371 | 0.99803 ± 0.00001 | 0.9075 ± 0.0031 | -0.0169 ± 0.0003 |
| PRIVATE | 20033 | 0.406 | 0.99804 ± 0.00001 | 0.9061 ± 0.0021 | -0.0195 ± 0.0002 |

### 2.3 Is the gain of W8P20 specific to the view? (same-view control)

| base | admixture (w 0.2) | val FINAL vs base | pmix-private | SVF | stale-val TO | holdout TO |
|---|---|---|---|---|---|---|
| W8 | Pd | +0.00049 ± 0.00031 | +0.00079 ± 0.00048 | +0.00064 ± 0.00033 | +0.00088 ± 0.00036 | -0.00063 ± 0.00022 |
| W8 | Pw | +0.00053 ± 0.00031 | +0.00057 ± 0.00048 | +0.00062 ± 0.00032 | +0.00091 ± 0.00034 | -0.00050 ± 0.00022 |
| W8 | Px8 | +0.00059 ± 0.00030 | +0.00056 ± 0.00047 | +0.00050 ± 0.00032 | +0.00086 ± 0.00034 | -0.00043 ± 0.00022 |
| W8 | W8b (same view: the other 8 wide students) | -0.00005 ± 0.00022 | +0.00042 ± 0.00033 | -0.00025 ± 0.00021 | -0.00014 ± 0.00021 | -0.00008 ± 0.00012 |
| W8 | Pd MINUS same view | +0.00053 ± 0.00033 | +0.00037 ± 0.00050 | +0.00089 ± 0.00033 | +0.00102 ± 0.00037 | -0.00054 ± 0.00022 |
| W8 | Pw MINUS same view | +0.00058 ± 0.00033 | +0.00015 ± 0.00051 | +0.00087 ± 0.00032 | +0.00105 ± 0.00035 | -0.00042 ± 0.00022 |
| W8 | Px8 MINUS same view | +0.00064 ± 0.00032 | +0.00014 ± 0.00049 | +0.00076 ± 0.00032 | +0.00100 ± 0.00036 | -0.00035 ± 0.00022 |
| W8b | Pd | +0.00079 ± 0.00033 | +0.00106 ± 0.00049 | +0.00066 ± 0.00031 | +0.00032 ± 0.00032 | -0.00049 ± 0.00022 |
| W8b | Pw | +0.00103 ± 0.00032 | +0.00143 ± 0.00047 | +0.00050 ± 0.00031 | +0.00072 ± 0.00032 | -0.00016 ± 0.00022 |
| W8b | Px8 | +0.00089 ± 0.00031 | +0.00124 ± 0.00045 | +0.00058 ± 0.00030 | +0.00052 ± 0.00031 | -0.00026 ± 0.00022 |
| W8b | W8 (same view: the other 8 wide students) | -0.00003 ± 0.00019 | +0.00029 ± 0.00030 | -0.00031 ± 0.00018 | -0.00009 ± 0.00020 | +0.00020 ± 0.00012 |
| W8b | Pd MINUS same view | +0.00082 ± 0.00032 | +0.00077 ± 0.00047 | +0.00096 ± 0.00030 | +0.00041 ± 0.00033 | -0.00068 ± 0.00022 |
| W8b | Pw MINUS same view | +0.00106 ± 0.00030 | +0.00114 ± 0.00044 | +0.00081 ± 0.00030 | +0.00081 ± 0.00032 | -0.00035 ± 0.00022 |
| W8b | Px8 MINUS same view | +0.00092 ± 0.00030 | +0.00095 ± 0.00043 | +0.00088 ± 0.00029 | +0.00061 ± 0.00032 | -0.00045 ± 0.00022 |

Reading [M]: 20% of fresh same-view members add nothing to an 8-pool at FINAL level (-0.00005, -0.00003); the popularity-light members add +0.0005 .. +0.0010 on BOTH disjoint 8-bases; view minus same view +0.00053 .. +0.00106 on val (z 1.6 - 3.5), +0.0008 .. +0.0010 on SVF. So the admixture gain is not "12 members instead of 8". On the 16-base the same admixture reads lower on fresh val (+0.00014 Pd, +0.00031 Px8, +0.00056 Pw) while stale-val and SVF keep +0.0006 .. +0.0010: the 16-base file is the average of the two 8-base twins and reads 0.0004 below both (-0.00044 ± 0.00021; a 2 SE oddity; I verified that the 16-base file equals the average of the twins branch by branch, so it is not a code error). I treat the differences BETWEEN popularity-light compositions at weight 0.2 as unresolved (all within 1.5 SE).

### 2.4 Compositions outside the short list (information; selection guard of R1)

| A - B | V1 val FINAL | V2 pmix-private | V3 SVF | centre c_val ± sd (sampling, spread) | PUBLIC ± SE | public shifted to private ± sd | **FORECAST private delta ± sd** | 90% interval of the REALISED private delta | P(A > B) expected / realised |
|---|---|---|---|---|---|---|---|---|---|
| W8P20x8 - W8P20 | +0.00010 ± 0.00019 | -0.00023 ± 0.00029 | -0.00014 ± 0.00019 | -0.00009 ± 0.00033 (0.00029, 0.00017) | none |  | **-0.00009 ± 0.00033** | [-0.00069, +0.00052] | 0.40 / 0.41 |
| W16P20w - W8P20 | +0.00000 ± 0.00034 | -0.00002 ± 0.00050 | -0.00011 ± 0.00034 | -0.00004 ± 0.00050 (0.00050, 0.00006) | none |  | **-0.00004 ± 0.00050** | [-0.00099, +0.00090] | 0.47 / 0.47 |
| W16P30x8 - W8P20 | +0.00006 ± 0.00033 | +0.00014 ± 0.00048 | -0.00007 ± 0.00032 | +0.00004 ± 0.00049 (0.00048, 0.00011) | none |  | **+0.00004 ± 0.00049** | [-0.00087, +0.00096] | 0.53 / 0.53 |
| W16P40x8 - W8P20 | +0.00030 ± 0.00037 | +0.00051 ± 0.00057 | +0.00023 ± 0.00036 | +0.00035 ± 0.00059 (0.00057, 0.00014) | none |  | **+0.00035 ± 0.00059** | [-0.00076, +0.00145] | 0.72 / 0.70 |
| W16P30d - W8P20 | -0.00012 ± 0.00032 | -0.00026 ± 0.00047 | +0.00009 ± 0.00031 | -0.00010 ± 0.00050 (0.00047, 0.00017) | none |  | **-0.00010 ± 0.00050** | [-0.00102, +0.00083] | 0.42 / 0.43 |
| W16P40d - W8P20 | -0.00007 ± 0.00038 | -0.00033 ± 0.00057 | +0.00017 ± 0.00037 | -0.00008 ± 0.00062 (0.00057, 0.00025) | none |  | **-0.00008 ± 0.00062** | [-0.00123, +0.00108] | 0.45 / 0.46 |
| W16P40w - W8P20 | +0.00022 ± 0.00040 | +0.00034 ± 0.00061 | +0.00007 ± 0.00039 | +0.00021 ± 0.00063 (0.00061, 0.00014) | none |  | **+0.00021 ± 0.00063** | [-0.00096, +0.00139] | 0.63 / 0.62 |
| W8P30x8 - W8P20 | +0.00002 ± 0.00026 | +0.00003 ± 0.00041 | -0.00024 ± 0.00024 | -0.00006 ± 0.00043 (0.00041, 0.00014) | none |  | **-0.00006 ± 0.00043** | [-0.00087, +0.00075] | 0.44 / 0.45 |
| W8P40x8 - W8P20 | +0.00004 ± 0.00032 | +0.00023 ± 0.00050 | -0.00010 ± 0.00031 | +0.00006 ± 0.00053 (0.00050, 0.00016) | none |  | **+0.00006 ± 0.00053** | [-0.00093, +0.00104] | 0.54 / 0.54 |
| W8P40 - W8P20 | -0.00019 ± 0.00034 | -0.00013 ± 0.00053 | +0.00006 ± 0.00032 | -0.00008 ± 0.00054 (0.00053, 0.00012) | none |  | **-0.00008 ± 0.00054** | [-0.00110, +0.00093] | 0.44 / 0.45 |
| S16 - W8P20 | -0.00041 ± 0.00038 | -0.00117 ± 0.00057 | -0.00072 ± 0.00039 | -0.00076 ± 0.00068 (0.00057, 0.00038) | -0.000931 ± 0.00050 | -0.00125 ± 0.00059 | **-0.00104 ± 0.00045** | [-0.00195, -0.00014] | 0.01 / 0.03 |
| C3 - W8P20 | -0.00113 ± 0.00051 | -0.00114 ± 0.00079 | -0.00155 ± 0.00051 | -0.00127 ± 0.00082 (0.00079, 0.00021) | -0.001555 ± 0.00066 | -0.00160 ± 0.00066 | **-0.00147 ± 0.00052** | [-0.00259, -0.00034] | 0.00 / 0.02 |
| RULEM1040 - W8P20 | +0.00068 ± 0.00035 | +0.00117 ± 0.00055 | +0.00015 ± 0.00033 | +0.00066 ± 0.00074 (0.00055, 0.00051) | none |  | **+0.00066 ± 0.00074** | [-0.00066, +0.00199] | 0.81 / 0.80 |
| RULEM2040 - W8P20 | +0.00039 ± 0.00035 | +0.00079 ± 0.00054 | +0.00016 ± 0.00033 | +0.00045 ± 0.00063 (0.00054, 0.00032) | none |  | **+0.00045 ± 0.00063** | [-0.00070, +0.00159] | 0.76 / 0.74 |
| RULEM1040w8 - W8P20 | +0.00047 ± 0.00028 | +0.00057 ± 0.00045 | +0.00033 ± 0.00028 | +0.00046 ± 0.00046 (0.00045, 0.00012) | none |  | **+0.00046 ± 0.00046** | [-0.00041, +0.00132] | 0.84 / 0.81 |
| RULEM1040d - W8P20 | +0.00023 ± 0.00028 | +0.00019 ± 0.00046 | +0.00029 ± 0.00028 | +0.00024 ± 0.00046 (0.00046, 0.00005) | none |  | **+0.00024 ± 0.00046** | [-0.00064, +0.00111] | 0.69 / 0.67 |
| RULEM2040w - W8P20 | +0.00071 ± 0.00037 | +0.00091 ± 0.00057 | +0.00024 ± 0.00036 | +0.00062 ± 0.00066 (0.00057, 0.00034) | none |  | **+0.00062 ± 0.00066** | [-0.00059, +0.00182] | 0.83 / 0.80 |

Nested picks (the out-of-sample value of choosing on val):

| list | in-sample best (selection-biased) | out-of-sample val FINAL vs W8P20 | out-of-sample pmix-private | out-of-sample SVF | picks |
|---|---|---|---|---|---|
| short list (5 files) | W8P20 (+0.00000 in c_val over W8P20) | -0.00013 (sd 0.00019) | -0.00020 (sd 0.00030) | -0.00012 (sd 0.00020) | W8P20 83%, W16P20x8 15%, W8 1%, W16P20d 0%, W16 0% |
| all weights and pools (20 files) | W16P40x8 (+0.00035 in c_val over W8P20) | -0.00002 (sd 0.00024) | -0.00002 (sd 0.00036) | -0.00007 (sd 0.00022) | W16P40x8 48%, W16P40w 20%, W8P20 8%, W16P20w 6%, W8P40x8 4% |
| W8P20 or W16P40x8 (2 files) | W16P40x8 (+0.00035 in c_val over W8P20) | +0.00018 (sd 0.00019) | +0.00032 (sd 0.00029) | +0.00013 (sd 0.00015) | W16P40x8 84%, W8P20 16% |
| W8P20 or the feed-size rule(s) (6 files) | RULEM1040 (+0.00066 in c_val over W8P20) | +0.00050 (sd 0.00017) | +0.00077 (sd 0.00027) | +0.00010 (sd 0.00012) | RULEM1040 47%, RULEM2040w 36%, RULEM1040w8 14%, RULEM1040d 3%, RULEM2040 1% |

Base-seed draw (the same admixture on W8, W8b, W16):

| admixture - its base | base W8 (s951-954): val FINAL / pmix-private / SVF / stale TO | base W8b (s955-958) | base W16 |
|---|---|---|---|
| 0.2 Pd | +0.00049 ± 0.00031 / +0.00079 / +0.00064 / +0.00088 | +0.00079 ± 0.00033 / +0.00106 / +0.00066 / +0.00032 | +0.00014 ± 0.00031 / +0.00004 / +0.00072 / +0.00081 |
| 0.2 Px8 | +0.00059 ± 0.00030 / +0.00056 / +0.00050 / +0.00086 | +0.00089 ± 0.00031 / +0.00124 / +0.00058 / +0.00052 | +0.00031 ± 0.00031 / +0.00023 / +0.00075 / +0.00065 |
| 0.2 Pw | +0.00053 ± 0.00031 / +0.00057 / +0.00062 / +0.00091 | +0.00103 ± 0.00032 / +0.00143 / +0.00050 / +0.00072 | +0.00056 ± 0.00032 / +0.00061 / +0.00100 / +0.00074 |
| 0.4 Pd | +0.00030 ± 0.00047 / +0.00066 / +0.00070 / +0.00121 | +0.00069 ± 0.00048 / +0.00087 / +0.00111 / +0.00081 | +0.00049 ± 0.00047 / +0.00031 / +0.00127 / +0.00122 |
| 0.4 Px8 | +0.00053 ± 0.00044 / +0.00102 / +0.00054 / +0.00125 | +0.00102 ± 0.00046 / +0.00130 / +0.00130 / +0.00128 | +0.00086 ± 0.00044 / +0.00115 / +0.00133 / +0.00147 |
| 0.4 Pw | +0.00046 ± 0.00043 / +0.00109 / +0.00074 / +0.00148 | +0.00142 ± 0.00046 / +0.00188 / +0.00127 / +0.00126 | +0.00078 ± 0.00044 / +0.00098 / +0.00117 / +0.00138 |
W8P20 minus its W8b twin (0.8 z(W8b) + 0.2 z(Pd)): val FINAL -0.00004 ± 0.00044, pmix-private +0.00027 ± 0.00065, SVF +0.00050 ± 0.00043. (W8 - W8b: val FINAL +0.00026 ± 0.00048.)

## 3. The popularity-light weight for private

### 3.1 Constant weight (rule R2)

| base + pop pool | w | val FINAL | val M>=50 | val M<50 | pmix-public | pmix-private (z) | stale-val TO | SVF | holdout TO (reported) | R2 without lagged clause |
|---|---|---|---|---|---|---|---|---|---|---|
| W8 + Pd (scored form) | 0.0 | -0.00049 ± 0.00031 | -0.00080 | -0.00003 | -0.00063 ± 0.00039 | -0.00079 ± 0.00048 (-1.63) | -0.00088 ± 0.00036 | -0.00064 ± 0.00033 | +0.00063 ± 0.00022 | no |
| W8 + Pd (scored form) | 0.1 | -0.00019 ± 0.00022 | -0.00037 | +0.00006 | -0.00031 ± 0.00028 | -0.00037 ± 0.00035 (-1.06) | -0.00051 ± 0.00026 | -0.00018 ± 0.00024 | +0.00035 ± 0.00015 | no |
| W8 + Pd (scored form) | 0.3 | -0.00036 ± 0.00023 | +0.00006 | -0.00098 | -0.00039 ± 0.00030 | -0.00039 ± 0.00037 (-1.04) | -0.00011 ± 0.00026 | +0.00017 ± 0.00023 | -0.00041 ± 0.00015 | no |
| W8 + Pd (scored form) | 0.4 | -0.00019 ± 0.00034 | +0.00035 | -0.00096 | -0.00025 ± 0.00043 | -0.00013 ± 0.00053 (-0.24) | +0.00033 ± 0.00035 | +0.00006 ± 0.00032 | -0.00108 ± 0.00023 | no |
| W8 + Px8 | 0.0 | -0.00059 ± 0.00030 | -0.00092 | -0.00012 | -0.00049 ± 0.00038 | -0.00056 ± 0.00047 (-1.18) | -0.00086 ± 0.00034 | -0.00050 ± 0.00032 | +0.00043 ± 0.00022 | no |
| W8 + Px8 | 0.1 | -0.00020 ± 0.00021 | -0.00032 | -0.00002 | -0.00019 ± 0.00027 | -0.00025 ± 0.00034 (-0.74) | -0.00059 ± 0.00024 | +0.00003 ± 0.00022 | +0.00012 ± 0.00015 | no |
| W8 + Px8 | 0.3 | -0.00008 ± 0.00022 | +0.00035 | -0.00071 | +0.00013 ± 0.00030 | +0.00026 ± 0.00037 (+0.71) | +0.00026 ± 0.00024 | -0.00011 ± 0.00022 | -0.00059 ± 0.00015 | no |
| W8 + Px8 | 0.4 | -0.00006 ± 0.00031 | +0.00063 | -0.00106 | +0.00023 ± 0.00039 | +0.00045 ± 0.00049 (+0.94) | +0.00039 ± 0.00032 | +0.00003 ± 0.00030 | -0.00100 ± 0.00021 | no |
| W16 + Pd | 0.0 | -0.00014 ± 0.00031 | -0.00076 | +0.00076 | -0.00006 ± 0.00039 | -0.00004 ± 0.00048 (-0.09) | -0.00081 ± 0.00035 | -0.00072 ± 0.00030 | +0.00056 ± 0.00022 | no |
| W16 + Pd | 0.1 | +0.00007 ± 0.00022 | -0.00048 | +0.00086 | +0.00018 ± 0.00028 | +0.00021 ± 0.00034 (+0.61) | -0.00027 ± 0.00025 | -0.00031 ± 0.00022 | +0.00032 ± 0.00015 | no |
| W16 + Pd | 0.3 | +0.00031 ± 0.00023 | +0.00065 | -0.00020 | +0.00023 ± 0.00028 | +0.00034 ± 0.00035 (+0.97) | +0.00003 ± 0.00023 | +0.00047 ± 0.00023 | -0.00073 ± 0.00016 | no |
| W16 + Pd | 0.4 | +0.00035 ± 0.00033 | +0.00073 | -0.00019 | +0.00015 ± 0.00041 | +0.00027 ± 0.00051 (+0.52) | +0.00041 ± 0.00034 | +0.00055 ± 0.00033 | -0.00120 ± 0.00023 | no |
| W16 + Px8 | 0.0 | -0.00031 ± 0.00031 | -0.00096 | +0.00062 | -0.00022 ± 0.00038 | -0.00023 ± 0.00046 (-0.49) | -0.00065 ± 0.00033 | -0.00075 ± 0.00029 | +0.00055 ± 0.00021 | no |
| W16 + Px8 | 0.1 | -0.00008 ± 0.00021 | -0.00063 | +0.00070 | +0.00014 ± 0.00026 | +0.00021 ± 0.00031 (+0.68) | -0.00029 ± 0.00023 | -0.00028 ± 0.00021 | +0.00038 ± 0.00015 | no |
| W16 + Px8 | 0.3 | +0.00030 ± 0.00021 | +0.00065 | -0.00020 | +0.00039 ± 0.00026 | +0.00056 ± 0.00031 (+1.77) | +0.00041 ± 0.00023 | +0.00028 ± 0.00022 | -0.00052 ± 0.00015 | PASS |
| W16 + Px8 | 0.4 | +0.00055 ± 0.00031 | +0.00107 | -0.00021 | +0.00066 ± 0.00040 | +0.00092 ± 0.00049 (+1.86) | +0.00082 ± 0.00033 | +0.00059 ± 0.00031 | -0.00111 ± 0.00022 | PASS |
| W16 + Pw (information) | 0.0 | -0.00056 ± 0.00032 | -0.00078 | -0.00025 | -0.00051 ± 0.00039 | -0.00061 ± 0.00048 (-1.28) | -0.00074 ± 0.00033 | -0.00100 ± 0.00031 | +0.00046 ± 0.00022 | no |
| W16 + Pw (information) | 0.1 | -0.00044 ± 0.00022 | -0.00050 | -0.00036 | -0.00035 ± 0.00028 | -0.00037 ± 0.00034 (-1.08) | -0.00035 ± 0.00022 | -0.00041 ± 0.00022 | +0.00028 ± 0.00015 | no |
| W16 + Pw (information) | 0.3 | +0.00022 ± 0.00020 | +0.00088 | -0.00074 | +0.00011 ± 0.00025 | +0.00004 ± 0.00030 (+0.14) | +0.00017 ± 0.00023 | +0.00002 ± 0.00021 | -0.00046 ± 0.00015 | no |
| W16 + Pw (information) | 0.4 | +0.00022 ± 0.00030 | +0.00119 | -0.00119 | +0.00026 ± 0.00037 | +0.00037 ± 0.00046 (+0.79) | +0.00065 ± 0.00034 | +0.00018 ± 0.00030 | -0.00090 ± 0.00022 | no |
Lagged holdout on record (W8 + Pd, train-only, private-like view, subset where the gate holds) [D]: S_lag(0.1) +0.00013, S_lag(0.2) +0.00024, S_lag(0.3) -0.00015, S_lag(0.4) -0.00053 (SE 0.00021-0.00045 each against W8): any w > 0.2 is contradicted there; w = 0.1 reads -0.00011 against 0.2.

Result by the rule as written. For the scored form (W8 + Pd) no weight beats 0.2 on any FINAL-level column; 0.2 is also the nested pick (54%). On W16 + Px8 the weights 0.3 and 0.4 pass R2 WITHOUT the lagged clause (0.4 against 0.2: val +0.00055 ± 0.00031, pmix-private +0.00092 ± 0.00049, stale-val +0.00082 ± 0.00033); with the lagged clause see 3.4. 12 declared looks + 4 on W16 + Pw.

### 3.2 Why a weight could depend on the feed: the level gap between the pools by feed group

| pools | group | feeds | share of val / public / private feeds | val FINAL | val TO | stale-val TO | val FT branch |
|---|---|---|---|---|---|---|---|
| Pd - W8 | M < 50 | 5893 | 0.409 / 0.387 / 0.392 | -0.00208 ± 0.00145 | -0.00250 ± 0.00155 | -0.00100 ± 0.00162 | -0.00184 ± 0.00142 |
| Pd - W8 | M >= 50 | 8530 | 0.591 / 0.613 / 0.608 | -0.00089 ± 0.00091 | -0.00188 ± 0.00103 | -0.00056 ± 0.00101 | +0.00036 ± 0.00088 |
| Pd - W8 | cold share < 0.30 | 8133 | 0.564 / 0.258 / 0.158 | -0.00082 ± 0.00106 | -0.00167 ± 0.00119 | -0.00051 ± 0.00121 | -0.00002 ± 0.00102 |
| Pd - W8 | cold share 0.30-0.45 | 5077 | 0.352 / 0.527 / 0.514 | -0.00292 ± 0.00132 | -0.00438 ± 0.00142 | -0.00257 ± 0.00142 | -0.00168 ± 0.00129 |
| Pd - W8 | cold share >= 0.45 | 1213 | 0.084 / 0.215 / 0.327 | +0.00133 ± 0.00319 | +0.00417 ± 0.00330 | +0.00535 ± 0.00334 | +0.00071 ± 0.00311 |
| Px8 - W16 | M < 50 | 5893 | 0.409 / 0.387 / 0.392 | -0.00294 ± 0.00136 | -0.00276 ± 0.00147 | -0.00135 ± 0.00154 | -0.00170 ± 0.00125 |
| Px8 - W16 | M >= 50 | 8530 | 0.591 / 0.613 / 0.608 | +0.00094 ± 0.00089 | -0.00074 ± 0.00100 | +0.00040 ± 0.00098 | +0.00173 ± 0.00083 |
| Px8 - W16 | cold share < 0.30 | 8133 | 0.564 / 0.258 / 0.158 | +0.00030 ± 0.00101 | -0.00153 ± 0.00113 | -0.00043 ± 0.00115 | +0.00037 ± 0.00094 |
| Px8 - W16 | cold share 0.30-0.45 | 5077 | 0.352 / 0.527 / 0.514 | -0.00222 ± 0.00128 | -0.00298 ± 0.00138 | -0.00173 ± 0.00139 | +0.00005 ± 0.00115 |
| Px8 - W16 | cold share >= 0.45 | 1213 | 0.084 / 0.215 / 0.327 | -0.00034 ± 0.00298 | +0.00410 ± 0.00311 | +0.00637 ± 0.00302 | +0.00121 ± 0.00290 |
| Pw - W16 | M < 50 | 5893 | 0.409 / 0.387 / 0.392 | -0.00417 ± 0.00137 | -0.00338 ± 0.00147 | -0.00211 ± 0.00155 | -0.00190 ± 0.00128 |
| Pw - W16 | M >= 50 | 8530 | 0.591 / 0.613 / 0.608 | +0.00150 ± 0.00090 | -0.00009 ± 0.00102 | +0.00150 ± 0.00101 | +0.00180 ± 0.00086 |
| Pw - W16 | cold share < 0.30 | 8133 | 0.564 / 0.258 / 0.158 | -0.00057 ± 0.00102 | -0.00157 ± 0.00114 | -0.00038 ± 0.00119 | +0.00044 ± 0.00097 |
| Pw - W16 | cold share 0.30-0.45 | 5077 | 0.352 / 0.527 / 0.514 | -0.00141 ± 0.00129 | -0.00200 ± 0.00142 | -0.00065 ± 0.00142 | -0.00033 ± 0.00118 |
| Pw - W16 | cold share >= 0.45 | 1213 | 0.084 / 0.215 / 0.327 | -0.00000 ± 0.00302 | +0.00185 ± 0.00304 | +0.00555 ± 0.00296 | +0.00185 ± 0.00296 |
Mean feed cold share: val 0.290, public 0.371, private 0.406. Share of feeds with M < 50: val 0.409, public 0.387, private 0.392.

On short feeds (M < 50; 39% of public and of private feeds) the popularity-light pool is 0.002 - 0.004 behind the base pool at every level; on long feeds it is within 1 SE of it or ahead at FINAL level (Pd -0.0009 ± 0.0009, Px8 +0.0009 ± 0.0009, Pw +0.0015 ± 0.0009) and level with it under stale inputs. Two pools of equal strength and correlation 0.98 want a weight near one half; a pool that is 0.003 behind wants a small one. That is the mechanism. The in-period holdout cannot see it: all its feeds are long AND its popularity is fresh, where the popularity-light pool is 0.006 - 0.007 behind.

### 3.3 Feed-dependent weight (rule R3: 3 properties x 25 weight pairs, nested on user halves)

**W8 + Pd (scored form)**
| in-sample best rule per property (selection-biased) | w (group false) | w (group true) | val FINAL vs 0.2 | pmix-private | stale-val TO |
|---|---|---|---|---|---|
| P_M (M >= 50) | 0.1 | 0.4 | +0.00023 ± 0.00028 | +0.00019 ± 0.00046 | +0.00023 ± 0.00030 |
| P_c30 (cold share >= 0.30) | 0.4 | 0.2 | +0.00018 ± 0.00024 | +0.00006 ± 0.00020 | -0.00007 ± 0.00027 |
| P_c45 (cold share >= 0.45) | 0.2 | 0.4 | +0.00010 ± 0.00012 | +0.00045 ± 0.00033 | +0.00024 ± 0.00011 |

- NESTED (75 rules): out-of-sample val FINAL vs constant 0.2 = **-0.00015** (sd over splits 0.00018; median per-split paired SE 0.00026; z -0.56); same picks under pmix-private weights -0.00018; on stale-val TO +0.00003. Most frequent picks: P_M 0.0/0.4 (19%); P_c30 0.4/0.2 (17%); P_M 0.1/0.4 (16%); P_M 0.2/0.4 (12%). **R3: not defensible -> constant weight.**
- NESTED, constant weights only (5 rules): out-of-sample val FINAL vs 0.2 = -0.00028 (sd 0.00021); pmix-private -0.00030; stale-val TO -0.00006; picks: 0.0: 3%, 0.1: 15%, 0.2: 54%, 0.3: 1%, 0.4: 26%.

**W16 + Px8**
| in-sample best rule per property (selection-biased) | w (group false) | w (group true) | val FINAL vs 0.2 | pmix-private | stale-val TO |
|---|---|---|---|---|---|
| P_M (M >= 50) | 0.1 | 0.4 | +0.00092 ± 0.00026 | +0.00158 ± 0.00044 | +0.00063 ± 0.00028 |
| P_c30 (cold share >= 0.30) | 0.4 | 0.1 | +0.00065 ± 0.00026 | +0.00039 ± 0.00034 | +0.00023 ± 0.00029 |
| P_c45 (cold share >= 0.45) | 0.4 | 0.4 | +0.00055 ± 0.00031 | +0.00092 ± 0.00049 | +0.00082 ± 0.00033 |

- NESTED (75 rules): out-of-sample val FINAL vs constant 0.2 = **+0.00065** (sd over splits 0.00020; median per-split paired SE 0.00028; z +2.31); same picks under pmix-private weights +0.00113; on stale-val TO +0.00058. Most frequent picks: P_M 0.1/0.4 (41%); P_M 0.0/0.4 (33%); P_c30 0.4/0.1 (8%); P_c30 0.4/0.3 (4%). **R3: DEFENSIBLE.**
- NESTED, constant weights only (5 rules): out-of-sample val FINAL vs 0.2 = +0.00034 (sd 0.00030); pmix-private +0.00068; stale-val TO +0.00063; picks: 0.0: 2%, 0.1: 5%, 0.2: 1%, 0.3: 11%, 0.4: 80%.

**W8 + Px8**
| in-sample best rule per property (selection-biased) | w (group false) | w (group true) | val FINAL vs 0.2 | pmix-private | stale-val TO |
|---|---|---|---|---|---|
| P_M (M >= 50) | 0.2 | 0.4 | +0.00037 ± 0.00021 | +0.00070 ± 0.00034 | +0.00024 ± 0.00022 |
| P_c30 (cold share >= 0.30) | 0.2 | 0.1 | +0.00007 ± 0.00014 | -0.00015 ± 0.00031 | -0.00004 ± 0.00014 |
| P_c45 (cold share >= 0.45) | 0.2 | 0.4 | +0.00020 ± 0.00011 | +0.00070 ± 0.00031 | +0.00011 ± 0.00010 |

- NESTED (75 rules): out-of-sample val FINAL vs constant 0.2 = **+0.00007** (sd over splits 0.00020; median per-split paired SE 0.00024; z +0.27); same picks under pmix-private weights +0.00042; on stale-val TO +0.00022. Most frequent picks: P_M 0.2/0.4 (28%); P_M 0.0/0.4 (25%); P_M 0.1/0.4 (18%); P_c45 0.2/0.4 (7%). **R3: not defensible -> constant weight.**
- NESTED, constant weights only (5 rules): out-of-sample val FINAL vs 0.2 = -0.00025 (sd 0.00014); pmix-private -0.00007; stale-val TO +0.00006; picks: 0.0: 1%, 0.1: 11%, 0.2: 40%, 0.3: 19%, 0.4: 30%.

**W16 + Pd**
| in-sample best rule per property (selection-biased) | w (group false) | w (group true) | val FINAL vs 0.2 | pmix-private | stale-val TO |
|---|---|---|---|---|---|
| P_M (M >= 50) | 0.1 | 0.4 | +0.00078 ± 0.00027 | +0.00084 ± 0.00044 | +0.00055 ± 0.00030 |
| P_c30 (cold share >= 0.30) | 0.4 | 0.1 | +0.00066 ± 0.00029 | +0.00033 ± 0.00038 | +0.00003 ± 0.00030 |
| P_c45 (cold share >= 0.45) | 0.3 | 0.4 | +0.00038 ± 0.00025 | +0.00049 ± 0.00042 | +0.00016 ± 0.00024 |

- NESTED (75 rules): out-of-sample val FINAL vs constant 0.2 = **+0.00047** (sd over splits 0.00017; median per-split paired SE 0.00028; z +1.64); same picks under pmix-private weights +0.00039; on stale-val TO +0.00024. Most frequent picks: P_M 0.1/0.4 (24%); P_M 0.0/0.4 (20%); P_M 0.1/0.3 (14%); P_c30 0.4/0.1 (13%). **R3: not defensible -> constant weight.**
- NESTED, constant weights only (5 rules): out-of-sample val FINAL vs 0.2 = +0.00009 (sd 0.00022); pmix-private +0.00008; stale-val TO +0.00007; picks: 0.0: 6%, 0.1: 11%, 0.2: 1%, 0.3: 30%, 0.4: 51%.

**W16 + Pw (information)**
| in-sample best rule per property (selection-biased) | w (group false) | w (group true) | val FINAL vs 0.2 | pmix-private | stale-val TO |
|---|---|---|---|---|---|
| P_M (M >= 50) | 0.2 | 0.4 | +0.00071 ± 0.00021 | +0.00093 ± 0.00034 | +0.00033 ± 0.00022 |
| P_c30 (cold share >= 0.30) | 0.4 | 0.2 | +0.00027 ± 0.00022 | +0.00002 ± 0.00018 | +0.00040 ± 0.00026 |
| P_c45 (cold share >= 0.45) | 0.3 | 0.4 | +0.00031 ± 0.00022 | +0.00052 ± 0.00038 | +0.00027 ± 0.00025 |

- NESTED (75 rules): out-of-sample val FINAL vs constant 0.2 = **+0.00045** (sd over splits 0.00020; median per-split paired SE 0.00026; z +1.74); same picks under pmix-private weights +0.00064; on stale-val TO +0.00037. Most frequent picks: P_M 0.2/0.4 (52%); P_M 0.0/0.4 (28%); P_M 0.1/0.4 (5%); P_M 0.2/0.3 (4%). **R3: not defensible -> constant weight.**
- NESTED, constant weights only (5 rules): out-of-sample val FINAL vs 0.2 = +0.00005 (sd 0.00019); pmix-private +0.00003; stale-val TO +0.00034; picks: 0.0: 2%, 0.1: 0%, 0.2: 10%, 0.3: 42%, 0.4: 46%.

**Summary over the five compositions (block 4)**
| composition | nested val | nested pmix-private | nested stale-val TO | z |
|---|---|---|---|---|
| W8 + Pd (scored form) | -0.00015 | -0.00018 | +0.00003 | -0.56 |
| W16 + Px8 | +0.00065 | +0.00113 | +0.00058 | +2.31 |
| W8 + Px8 | +0.00007 | +0.00042 | +0.00022 | +0.27 |
| W16 + Pd | +0.00047 | +0.00039 | +0.00024 | +1.64 |
| W16 + Pw (information) | +0.00045 | +0.00064 | +0.00037 | +1.74 |
Mean nested val delta +0.00030; compositions with all three columns >= 0: 4 of 5 -> "one composition only".

Result by the rule as written: **DEFENSIBLE on W16 + Px8** (nested out-of-sample +0.00065 over constant 0.2, z 2.31; pmix-private +0.00113; stale-val +0.00058), **not on the scored form W8 + Pd** (-0.00015). Over the five compositions the mean nested gain is +0.000298 with all three columns >= 0 in 4 of 5: by the letter of block 4 (bar +0.0003) "one composition only", by a hair. Multiplicity: 75 rules x 5 compositions (2 declared in block 1, 2 run with them, 1 declared in block 4); the modal property is feed size in all five, and the hypothesis "short feeds dislike the admixture" was formed on the full val sample (witness.md 3.4) before my nested read, so the nested z is optimistic. What is consistent across all five compositions is narrower than the rule: on M >= 50 the weight 0.4 beats 0.2 in 5 of 5 (+0.00035 .. +0.00119 per long feed), on M < 50 the weight 0.4 loses to 0.2 in 5 of 5 (-0.0002 .. -0.0012), and 0.1 against 0.2 on short feeds is mixed. Robustness of the rule files against the constant-0.2 file of the same members (in-sample, selection-biased; all slices) and what a public slot would read:

**T7 (val FINAL) by slice**
| A - B | all | Tue | Thu | M<50 | M>=50 | cold<0.30 | cold 0.30-0.45 | cold>=0.45 | user half A | user half B |
|---|---|---|---|---|---|---|---|---|---|---|
| RULEM1040 - W8P20 | +0.00068 ± 0.00035 | +0.00060 ± 0.00052 | +0.00076 ± 0.00048 | +0.00022 ± 0.00064 | +0.00099 ± 0.00040 | +0.00043 ± 0.00046 | +0.00098 ± 0.00059 | +0.00106 ± 0.00142 | +0.00057 ± 0.00051 | +0.00079 ± 0.00049 |
| RULEM2040 - W8P20 | +0.00039 ± 0.00035 | +0.00050 ± 0.00051 | +0.00027 ± 0.00047 | -0.00048 ± 0.00062 | +0.00099 ± 0.00040 | +0.00033 ± 0.00045 | +0.00033 ± 0.00058 | +0.00101 ± 0.00145 | +0.00048 ± 0.00051 | +0.00030 ± 0.00047 |
| W16P40x8 - W8P20 | +0.00030 ± 0.00037 | +0.00059 ± 0.00055 | -0.00001 ± 0.00050 | -0.00069 ± 0.00070 | +0.00099 ± 0.00040 | +0.00064 ± 0.00049 | -0.00036 ± 0.00062 | +0.00080 ± 0.00150 | +0.00033 ± 0.00055 | +0.00028 ± 0.00051 |
| W16P20x8 - W8P20 | -0.00025 ± 0.00032 | -0.00020 ± 0.00047 | -0.00030 ± 0.00044 | -0.00048 ± 0.00062 | -0.00008 ± 0.00034 | -0.00029 ± 0.00043 | +0.00005 ± 0.00052 | -0.00117 ± 0.00133 | -0.00044 ± 0.00048 | -0.00005 ± 0.00043 |
| RULEM1040 - W16 | +0.00123 ± 0.00035 | +0.00148 ± 0.00050 | +0.00097 ± 0.00048 | +0.00008 ± 0.00042 | +0.00203 ± 0.00051 | +0.00131 ± 0.00044 | +0.00094 ± 0.00061 | +0.00194 ± 0.00128 | +0.00136 ± 0.00049 | +0.00111 ± 0.00049 |
| RULEM2040 - W16 | +0.00095 ± 0.00038 | +0.00138 ± 0.00055 | +0.00048 ± 0.00052 | -0.00062 ± 0.00057 | +0.00203 ± 0.00051 | +0.00122 ± 0.00049 | +0.00029 ± 0.00066 | +0.00190 ± 0.00148 | +0.00127 ± 0.00054 | +0.00062 ± 0.00054 |
| W16P40x8 - W16P20x8 | +0.00055 ± 0.00031 | +0.00079 ± 0.00044 | +0.00028 ± 0.00042 | -0.00021 ± 0.00056 | +0.00107 ± 0.00035 | +0.00094 ± 0.00039 | -0.00041 ± 0.00054 | +0.00197 ± 0.00125 | +0.00077 ± 0.00045 | +0.00033 ± 0.00042 |
| RULEM1040 - W16P20x8 | +0.00092 ± 0.00026 | +0.00080 ± 0.00037 | +0.00106 ± 0.00036 | +0.00070 ± 0.00037 | +0.00107 ± 0.00035 | +0.00072 ± 0.00032 | +0.00093 ± 0.00044 | +0.00223 ± 0.00115 | +0.00101 ± 0.00038 | +0.00084 ± 0.00035 |
| RULEM2040 - W16P20x8 | +0.00064 ± 0.00021 | +0.00070 ± 0.00030 | +0.00056 ± 0.00029 | +0.00000 ± 0.00000 | +0.00107 ± 0.00035 | +0.00062 ± 0.00025 | +0.00028 ± 0.00038 | +0.00218 ± 0.00088 | +0.00092 ± 0.00031 | +0.00035 ± 0.00028 |
| RULEM2040w - W16P20w | +0.00071 ± 0.00021 | +0.00079 ± 0.00030 | +0.00061 ± 0.00028 | +0.00000 ± 0.00000 | +0.00119 ± 0.00035 | +0.00078 ± 0.00027 | +0.00046 ± 0.00036 | +0.00124 ± 0.00076 | +0.00076 ± 0.00031 | +0.00065 ± 0.00028 |
| W8P20 - W16 | +0.00056 ± 0.00038 | +0.00087 ± 0.00054 | +0.00021 ± 0.00053 | -0.00013 ± 0.00073 | +0.00104 ± 0.00040 | +0.00089 ± 0.00051 | -0.00004 ± 0.00063 | +0.00088 ± 0.00144 | +0.00079 ± 0.00054 | +0.00032 ± 0.00053 |
| W8P20 - W8 | +0.00049 ± 0.00031 | +0.00100 ± 0.00045 | -0.00008 ± 0.00043 | +0.00003 ± 0.00058 | +0.00080 ± 0.00035 | +0.00038 ± 0.00042 | +0.00046 ± 0.00049 | +0.00128 ± 0.00134 | +0.00021 ± 0.00044 | +0.00076 ± 0.00044 |

**T7 (stale-val TO) by slice**
| A - B | all | Tue | Thu | M<50 | M>=50 | cold<0.30 | cold 0.30-0.45 | cold>=0.45 | user half A | user half B |
|---|---|---|---|---|---|---|---|---|---|---|
| RULEM1040 - W8P20 | +0.00053 ± 0.00037 | +0.00079 ± 0.00052 | +0.00024 ± 0.00052 | +0.00064 ± 0.00068 | +0.00045 ± 0.00041 | +0.00000 ± 0.00051 | +0.00113 ± 0.00060 | +0.00151 ± 0.00116 | +0.00015 ± 0.00053 | +0.00090 ± 0.00051 |
| RULEM2040 - W8P20 | +0.00032 ± 0.00034 | +0.00039 ± 0.00049 | +0.00025 ± 0.00047 | +0.00014 ± 0.00059 | +0.00045 ± 0.00041 | -0.00019 ± 0.00048 | +0.00096 ± 0.00054 | +0.00107 ± 0.00107 | -0.00006 ± 0.00049 | +0.00070 ± 0.00048 |
| W16P40x8 - W8P20 | +0.00071 ± 0.00037 | +0.00078 ± 0.00053 | +0.00065 ± 0.00052 | +0.00110 ± 0.00069 | +0.00045 ± 0.00041 | +0.00025 ± 0.00051 | +0.00098 ± 0.00060 | +0.00274 ± 0.00128 | +0.00044 ± 0.00053 | +0.00099 ± 0.00052 |
| W16P20x8 - W8P20 | -0.00011 ± 0.00031 | -0.00003 ± 0.00046 | -0.00019 ± 0.00043 | +0.00014 ± 0.00059 | -0.00027 ± 0.00034 | -0.00041 ± 0.00045 | +0.00029 ± 0.00047 | +0.00027 ± 0.00107 | -0.00036 ± 0.00045 | +0.00015 ± 0.00044 |
| RULEM1040 - W16 | +0.00128 ± 0.00035 | +0.00139 ± 0.00051 | +0.00116 ± 0.00048 | -0.00039 ± 0.00043 | +0.00243 ± 0.00052 | +0.00127 ± 0.00047 | +0.00104 ± 0.00060 | +0.00236 ± 0.00117 | +0.00135 ± 0.00049 | +0.00121 ± 0.00051 |
| RULEM2040 - W16 | +0.00107 ± 0.00040 | +0.00099 ± 0.00058 | +0.00116 ± 0.00055 | -0.00090 ± 0.00063 | +0.00243 ± 0.00052 | +0.00108 ± 0.00053 | +0.00086 ± 0.00068 | +0.00192 ± 0.00130 | +0.00114 ± 0.00055 | +0.00101 ± 0.00058 |
| W16P40x8 - W16P20x8 | +0.00082 ± 0.00033 | +0.00080 ± 0.00045 | +0.00083 ± 0.00050 | +0.00096 ± 0.00062 | +0.00072 ± 0.00037 | +0.00066 ± 0.00044 | +0.00068 ± 0.00056 | +0.00247 ± 0.00124 | +0.00080 ± 0.00046 | +0.00084 ± 0.00049 |
| RULEM1040 - W16P20x8 | +0.00063 ± 0.00028 | +0.00081 ± 0.00040 | +0.00043 ± 0.00041 | +0.00050 ± 0.00044 | +0.00072 ± 0.00037 | +0.00041 ± 0.00038 | +0.00084 ± 0.00049 | +0.00124 ± 0.00081 | +0.00051 ± 0.00040 | +0.00075 ± 0.00040 |
| RULEM2040 - W16P20x8 | +0.00043 ± 0.00022 | +0.00041 ± 0.00031 | +0.00044 ± 0.00031 | +0.00000 ± 0.00000 | +0.00072 ± 0.00037 | +0.00022 ± 0.00029 | +0.00066 ± 0.00039 | +0.00080 ± 0.00059 | +0.00030 ± 0.00031 | +0.00055 ± 0.00031 |
| RULEM2040w - W16P20w | +0.00033 ± 0.00022 | +0.00042 ± 0.00032 | +0.00022 ± 0.00031 | +0.00000 ± 0.00000 | +0.00055 ± 0.00038 | +0.00016 ± 0.00028 | +0.00041 ± 0.00041 | +0.00110 ± 0.00063 | +0.00030 ± 0.00032 | +0.00035 ± 0.00031 |
| W8P20 - W16 | +0.00075 ± 0.00041 | +0.00061 ± 0.00059 | +0.00092 ± 0.00056 | -0.00103 ± 0.00079 | +0.00199 ± 0.00043 | +0.00127 ± 0.00056 | -0.00009 ± 0.00066 | +0.00085 ± 0.00148 | +0.00120 ± 0.00058 | +0.00031 ± 0.00058 |
| W8P20 - W8 | +0.00088 ± 0.00036 | +0.00097 ± 0.00050 | +0.00078 ± 0.00050 | -0.00034 ± 0.00066 | +0.00172 ± 0.00039 | +0.00114 ± 0.00049 | +0.00057 ± 0.00056 | +0.00040 ± 0.00128 | +0.00103 ± 0.00051 | +0.00073 ± 0.00050 |

**T7 (SVF) by slice**
| A - B | all | Tue | Thu | M<50 | M>=50 | cold<0.30 | cold 0.30-0.45 | cold>=0.45 | user half A | user half B |
|---|---|---|---|---|---|---|---|---|---|---|
| RULEM1040 - W8P20 | +0.00015 ± 0.00033 | -0.00009 ± 0.00047 | +0.00042 ± 0.00048 | -0.00048 ± 0.00063 | +0.00059 ± 0.00036 | +0.00005 ± 0.00044 | +0.00030 ± 0.00056 | +0.00024 ± 0.00123 | +0.00050 ± 0.00049 | -0.00019 ± 0.00046 |
| RULEM2040 - W8P20 | +0.00016 ± 0.00033 | +0.00014 ± 0.00044 | +0.00019 ± 0.00048 | -0.00046 ± 0.00060 | +0.00059 ± 0.00036 | +0.00013 ± 0.00043 | +0.00004 ± 0.00054 | +0.00082 ± 0.00121 | +0.00074 ± 0.00047 | -0.00042 ± 0.00046 |
| W16P40x8 - W8P20 | +0.00023 ± 0.00036 | +0.00047 ± 0.00051 | -0.00004 ± 0.00050 | -0.00029 ± 0.00070 | +0.00059 ± 0.00036 | +0.00051 ± 0.00047 | -0.00043 ± 0.00060 | +0.00112 ± 0.00136 | +0.00064 ± 0.00050 | -0.00018 ± 0.00051 |
| W16P20x8 - W8P20 | -0.00036 ± 0.00031 | -0.00053 ± 0.00043 | -0.00017 ± 0.00046 | -0.00046 ± 0.00060 | -0.00028 ± 0.00033 | -0.00061 ± 0.00041 | +0.00004 ± 0.00050 | -0.00031 ± 0.00127 | -0.00039 ± 0.00045 | -0.00033 ± 0.00044 |
| RULEM1040 - W16 | +0.00126 ± 0.00033 | +0.00114 ± 0.00047 | +0.00139 ± 0.00046 | +0.00020 ± 0.00039 | +0.00199 ± 0.00048 | +0.00132 ± 0.00044 | +0.00105 ± 0.00054 | +0.00171 ± 0.00116 | +0.00218 ± 0.00047 | +0.00033 ± 0.00045 |
| RULEM2040 - W16 | +0.00126 ± 0.00036 | +0.00136 ± 0.00051 | +0.00116 ± 0.00051 | +0.00022 ± 0.00054 | +0.00199 ± 0.00048 | +0.00140 ± 0.00049 | +0.00079 ± 0.00058 | +0.00230 ± 0.00130 | +0.00242 ± 0.00052 | +0.00011 ± 0.00050 |
| W16P40x8 - W16P20x8 | +0.00059 ± 0.00031 | +0.00100 ± 0.00045 | +0.00013 ± 0.00042 | +0.00017 ± 0.00059 | +0.00087 ± 0.00034 | +0.00112 ± 0.00040 | -0.00046 ± 0.00052 | +0.00143 ± 0.00139 | +0.00103 ± 0.00045 | +0.00014 ± 0.00043 |
| RULEM1040 - W16P20x8 | +0.00051 ± 0.00025 | +0.00044 ± 0.00035 | +0.00058 ± 0.00035 | -0.00002 ± 0.00036 | +0.00087 ± 0.00034 | +0.00066 ± 0.00033 | +0.00026 ± 0.00040 | +0.00054 ± 0.00101 | +0.00088 ± 0.00039 | +0.00013 ± 0.00031 |
| RULEM2040 - W16P20x8 | +0.00052 ± 0.00020 | +0.00066 ± 0.00029 | +0.00035 ± 0.00027 | +0.00000 ± 0.00000 | +0.00087 ± 0.00034 | +0.00074 ± 0.00025 | +0.00001 ± 0.00034 | +0.00113 ± 0.00087 | +0.00113 ± 0.00031 | -0.00010 ± 0.00024 |
| RULEM2040w - W16P20w | +0.00034 ± 0.00020 | +0.00038 ± 0.00029 | +0.00030 ± 0.00028 | +0.00000 ± 0.00000 | +0.00058 ± 0.00034 | +0.00041 ± 0.00025 | +0.00014 ± 0.00036 | +0.00075 ± 0.00081 | +0.00074 ± 0.00030 | -0.00005 ± 0.00026 |
| W8P20 - W16 | +0.00110 ± 0.00038 | +0.00122 ± 0.00053 | +0.00097 ± 0.00056 | +0.00068 ± 0.00072 | +0.00140 ± 0.00042 | +0.00127 ± 0.00051 | +0.00075 ± 0.00064 | +0.00148 ± 0.00141 | +0.00168 ± 0.00054 | +0.00053 ± 0.00054 |
| W8P20 - W8 | +0.00064 ± 0.00033 | +0.00101 ± 0.00047 | +0.00023 ± 0.00048 | -0.00049 ± 0.00062 | +0.00142 ± 0.00036 | +0.00069 ± 0.00043 | +0.00027 ± 0.00055 | +0.00188 ± 0.00146 | +0.00051 ± 0.00045 | +0.00077 ± 0.00049 |

**T7b. What a public slot would read (expected public delta = pmix-public of the val FINAL delta; public SE = sd of the per-feed delta / sqrt(8,562))**
| A - B | val FINAL | pmix-public | pmix-private | public SE of the pair | holdout TO (reported) |
|---|---|---|---|---|---|
| RULEM1040 - W8P20 | +0.00068 ± 0.00035 | +0.00103 ± 0.00044 | +0.00117 ± 0.00055 | 0.00046 | -0.00100 ± 0.00024 |
| RULEM2040 - W8P20 | +0.00039 ± 0.00035 | +0.00064 ± 0.00044 | +0.00079 ± 0.00054 | 0.00045 | -0.00100 ± 0.00024 |
| W16P40x8 - W8P20 | +0.00030 ± 0.00037 | +0.00040 ± 0.00047 | +0.00051 ± 0.00057 | 0.00048 | -0.00100 ± 0.00024 |
| W16P30x8 - W8P20 | +0.00006 ± 0.00033 | +0.00012 ± 0.00040 | +0.00014 ± 0.00048 | 0.00043 | -0.00040 ± 0.00020 |
| W16P20x8 - W8P20 | -0.00025 ± 0.00032 | -0.00027 ± 0.00039 | -0.00041 ± 0.00048 | 0.00042 | +0.00012 ± 0.00017 |
| W16P20w - W8P20 | +0.00000 ± 0.00034 | +0.00002 ± 0.00041 | -0.00002 ± 0.00050 | 0.00044 | +0.00021 ± 0.00018 |
| RULEM2040w - W8P20 | +0.00071 ± 0.00037 | +0.00083 ± 0.00046 | +0.00091 ± 0.00057 | 0.00048 | -0.00069 ± 0.00025 |
| RULEM1040 - W16 | +0.00123 ± 0.00035 | +0.00152 ± 0.00045 | +0.00181 ± 0.00056 | 0.00045 | -0.00166 ± 0.00032 |
| RULEM2040 - W16 | +0.00095 ± 0.00038 | +0.00113 ± 0.00049 | +0.00143 ± 0.00061 | 0.00049 | -0.00166 ± 0.00032 |
| W16P40x8 - W16 | +0.00086 ± 0.00044 | +0.00088 ± 0.00056 | +0.00115 ± 0.00070 | 0.00058 | -0.00166 ± 0.00032 |
| W16P20x8 - W16 | +0.00031 ± 0.00031 | +0.00022 ± 0.00038 | +0.00023 ± 0.00046 | 0.00040 | -0.00055 ± 0.00021 |
| RULEM1040 - W16P20x8 | +0.00092 ± 0.00026 | +0.00130 ± 0.00034 | +0.00158 ± 0.00044 | 0.00033 | -0.00111 ± 0.00022 |
| RULEM2040 - W16P20x8 | +0.00064 ± 0.00021 | +0.00091 ± 0.00029 | +0.00120 ± 0.00037 | 0.00027 | -0.00111 ± 0.00022 |
| RULEM2040w - W16P20w | +0.00071 ± 0.00021 | +0.00081 ± 0.00027 | +0.00093 ± 0.00034 | 0.00027 | -0.00090 ± 0.00022 |

### 3.4 The lagged holdout (my read; PREDECLARED block 3)

Tool control [M]: `abl_e10_pop1kd05t25_s951`, private-like view, re-scored by my driver against the witness cache: max |logit| difference 1.53e-05 (bar 1e-4): VALID. My read reproduces the record for W8 + Pd exactly (S_lag 0.1 / 0.2 / 0.3 / 0.4 = +0.00013 / +0.00024 / -0.00015 / -0.00053). New members scored by me (lag 0 and private-like, 2.3 - 3.3 CPU min per view): the 4 wide popularity-light students (Pw) and the 8 newer wide students (W8b; with them the W16 base). That gives 3 bases x 3 popularity-light pools.

Result [M].
1. **Weight 0.4 against 0.2 on the lagged holdout is negative in 9 of 9 compositions**: -0.00017 .. -0.00077 (SE 0.00028 - 0.00032 each); mean of the nine per feed -0.00045 ± 0.00023 (-2.0 SE); the declared composition W16 + Px8 reads -0.00043 ± 0.00029 (-1.5 SE). Weight 0.3 against 0.2: -0.00014 .. -0.00039. Weight 0.1 against 0.2: -0.00011 .. +0.00006 (the lagged holdout cannot tell 0.1 from 0.2). By block 3: the lagged clause of R2 FAILS for every weight above 0.2, and the feed-size rule of R3 is **"val-only, contradicted on the lagged holdout"** on its own pools.
2. **At the scored weight 0.2 the lagged holdout reads the admixture at zero, not at a gain**: S_lag(0.2) over the nine compositions -0.00014 .. +0.00024, mean +0.00003 ± 0.00023 (in-period lag 0: mean -0.00039 ± 0.00027). The +0.00024 on record (W8 + Pd) is the most favourable of the nine; its W8b twin reads -0.00008 ± 0.00030 with D = +0.00015 ± 0.00039 (W8: D = +0.00074 ± 0.00040); the composition with no member in common with the record (W8b + Pw) reads -0.00000 ± 0.00029. So "the bias explanation holds" survives in the weak form the reviewer gave it (the penalty disappears, it does not become a gain), and the 1.85 SE of D on record is a favourable base-seed draw: W8 loses 0.0033 under the lag, W8b 0.0022.
3. Under the private-like lag the train-only pool of W16 equals the train-only half of W8P20 (W16 - [W8 + 0.2 Pd] = +0.00000 ± 0.00034) and [W16 + 0.2 Px8] equals it too (-0.00001 ± 0.00024); [W16 + 0.4 Px8] is -0.00043 ± 0.00032 below it.
4. The popularity-light pool stays 0.0019 - 0.0030 behind the base pool under the lag (Px8 - W16: -0.00218 ± 0.00079; lag 0: -0.00517), where on the long feeds of val it is 0.0007 behind at train-only level on fresh inputs, level under stale-val inputs and ahead at FINAL level (section 3.2).

How I read the disagreement [I]. The lagged holdout reads this family about 0.0005 lower than the leaderboard does at weight 0.2 (0.0000 against +0.00052, +0.00057, +0.00070 on three public contrasts) for the reason written in block 3 before the read: its labels are in-period, so popularity counted inside the train window still predicts them and the full-popularity base keeps an advantage (item 4) that it does not have out of period. If that under-read grows with the weight, -0.0004 for 0.4 against 0.2 is what a true +0.0001 would look like. That is an argument, not a measurement: the lagged holdout is the only labelled witness of this question that does not share the val sample, and it says no.

Available: bases ['W8', 'W8b', 'W16']; popularity-light pools ['Pd', 'Pw', 'Px8'].

| pool | level lag 0 | level private-like | change |
|---|---|---|---|
| W8 | 0.32334 | 0.32002 | -0.00332 ± 0.00075 |
| W8b | 0.32233 | 0.32009 | -0.00224 ± 0.00074 |
| W16 | 0.32340 | 0.32026 | -0.00314 ± 0.00075 |
| Pd | 0.31773 | 0.31724 | -0.00049 ± 0.00038 |
| Pw | 0.31797 | 0.31773 | -0.00025 ± 0.00040 |
| Px8 | 0.31823 | 0.31808 | -0.00015 ± 0.00037 |

| pop pool - base pool | lag 0 | private-like |
|---|---|---|
| Pd - W8 | -0.00561 ± 0.00084 | -0.00278 ± 0.00082 |
| Pw - W8 | -0.00537 ± 0.00083 | -0.00229 ± 0.00082 |
| Px8 - W8 | -0.00511 ± 0.00082 | -0.00194 ± 0.00080 |
| Pd - W8b | -0.00460 ± 0.00084 | -0.00285 ± 0.00083 |
| Pw - W8b | -0.00436 ± 0.00084 | -0.00236 ± 0.00083 |
| Px8 - W8b | -0.00409 ± 0.00083 | -0.00200 ± 0.00081 |
| Pd - W16 | -0.00567 ± 0.00084 | -0.00302 ± 0.00081 |
| Pw - W16 | -0.00543 ± 0.00084 | -0.00254 ± 0.00081 |
| Px8 - W16 | -0.00517 ± 0.00082 | -0.00218 ± 0.00079 |

| base + pop | w | S_0(w) (lag 0) | S_lag(w) (private-like) | D = S_lag - S_0 | S_lag(w) - S_lag(0.2) | S_0(w) - S_0(0.2) |
|---|---|---|---|---|---|---|
| W8 + Pd | 0.1 | -0.00020 ± 0.00021 | +0.00013 ± 0.00021 | +0.00033 ± 0.00028 | -0.00011 ± 0.00021 | +0.00030 ± 0.00023 |
| W8 + Pd | 0.2 | -0.00050 ± 0.00032 | +0.00024 ± 0.00031 | +0.00074 ± 0.00040 |  |  |
| W8 + Pd | 0.3 | -0.00095 ± 0.00041 | -0.00015 ± 0.00038 | +0.00080 ± 0.00047 | -0.00039 ± 0.00020 | -0.00045 ± 0.00023 |
| W8 + Pd | 0.4 | -0.00144 ± 0.00048 | -0.00053 ± 0.00045 | +0.00090 ± 0.00053 | -0.00077 ± 0.00029 | -0.00094 ± 0.00033 |
| W8 + Pw | 0.1 | -0.00015 ± 0.00020 | +0.00002 ± 0.00020 | +0.00016 ± 0.00026 | +0.00001 ± 0.00021 | +0.00029 ± 0.00023 |
| W8 + Pw | 0.2 | -0.00044 ± 0.00032 | +0.00001 ± 0.00030 | +0.00045 ± 0.00038 |  |  |
| W8 + Pw | 0.3 | -0.00101 ± 0.00039 | -0.00013 ± 0.00037 | +0.00087 ± 0.00045 | -0.00014 ± 0.00020 | -0.00057 ± 0.00020 |
| W8 + Pw | 0.4 | -0.00129 ± 0.00046 | -0.00039 ± 0.00043 | +0.00090 ± 0.00051 | -0.00039 ± 0.00028 | -0.00085 ± 0.00031 |
| W8 + Px8 | 0.1 | -0.00032 ± 0.00021 | +0.00008 ± 0.00020 | +0.00040 ± 0.00026 | -0.00011 ± 0.00020 | +0.00011 ± 0.00022 |
| W8 + Px8 | 0.2 | -0.00043 ± 0.00031 | +0.00018 ± 0.00030 | +0.00061 ± 0.00038 |  |  |
| W8 + Px8 | 0.3 | -0.00103 ± 0.00039 | +0.00002 ± 0.00036 | +0.00105 ± 0.00045 | -0.00016 ± 0.00019 | -0.00060 ± 0.00021 |
| W8 + Px8 | 0.4 | -0.00125 ± 0.00046 | -0.00027 ± 0.00044 | +0.00099 ± 0.00051 | -0.00045 ± 0.00029 | -0.00083 ± 0.00031 |
| W8b + Pd | 0.1 | +0.00003 ± 0.00023 | -0.00008 ± 0.00020 | -0.00011 ± 0.00027 | +0.00000 ± 0.00021 | +0.00026 ± 0.00024 |
| W8b + Pd | 0.2 | -0.00023 ± 0.00034 | -0.00008 ± 0.00030 | +0.00015 ± 0.00039 |  |  |
| W8b + Pd | 0.3 | -0.00034 ± 0.00042 | -0.00043 ± 0.00040 | -0.00009 ± 0.00048 | -0.00035 ± 0.00023 | -0.00011 ± 0.00022 |
| W8b + Pd | 0.4 | -0.00068 ± 0.00050 | -0.00050 ± 0.00047 | +0.00018 ± 0.00054 | -0.00042 ± 0.00032 | -0.00045 ± 0.00032 |
| W8b + Pw | 0.1 | +0.00024 ± 0.00022 | +0.00005 ± 0.00020 | -0.00019 ± 0.00028 | +0.00005 ± 0.00019 | +0.00018 ± 0.00022 |
| W8b + Pw | 0.2 | +0.00006 ± 0.00032 | -0.00000 ± 0.00029 | -0.00006 ± 0.00038 |  |  |
| W8b + Pw | 0.3 | -0.00022 ± 0.00041 | -0.00023 ± 0.00039 | -0.00001 ± 0.00046 | -0.00023 ± 0.00023 | -0.00028 ± 0.00022 |
| W8b + Pw | 0.4 | -0.00035 ± 0.00048 | -0.00017 ± 0.00046 | +0.00018 ± 0.00052 | -0.00017 ± 0.00031 | -0.00041 ± 0.00031 |
| W8b + Px8 | 0.1 | +0.00030 ± 0.00023 | -0.00003 ± 0.00020 | -0.00033 ± 0.00028 | +0.00006 ± 0.00020 | +0.00039 ± 0.00023 |
| W8b + Px8 | 0.2 | -0.00009 ± 0.00033 | -0.00009 ± 0.00029 | +0.00000 ± 0.00039 |  |  |
| W8b + Px8 | 0.3 | -0.00032 ± 0.00041 | -0.00040 ± 0.00038 | -0.00008 ± 0.00046 | -0.00031 ± 0.00022 | -0.00023 ± 0.00020 |
| W8b + Px8 | 0.4 | -0.00038 ± 0.00047 | -0.00033 ± 0.00045 | +0.00005 ± 0.00052 | -0.00024 ± 0.00031 | -0.00029 ± 0.00030 |
| W16 + Pd | 0.1 | -0.00032 ± 0.00023 | -0.00012 ± 0.00020 | +0.00020 ± 0.00029 | +0.00002 ± 0.00022 | +0.00014 ± 0.00023 |
| W16 + Pd | 0.2 | -0.00046 ± 0.00034 | -0.00014 ± 0.00030 | +0.00032 ± 0.00041 |  |  |
| W16 + Pd | 0.3 | -0.00149 ± 0.00042 | -0.00036 ± 0.00038 | +0.00113 ± 0.00047 | -0.00021 ± 0.00020 | -0.00103 ± 0.00022 |
| W16 + Pd | 0.4 | -0.00165 ± 0.00049 | -0.00068 ± 0.00046 | +0.00098 ± 0.00054 | -0.00054 ± 0.00030 | -0.00120 ± 0.00032 |
| W16 + Pw | 0.1 | -0.00034 ± 0.00022 | +0.00017 ± 0.00020 | +0.00051 ± 0.00029 | -0.00002 ± 0.00020 | +0.00029 ± 0.00023 |
| W16 + Pw | 0.2 | -0.00063 ± 0.00033 | +0.00019 ± 0.00029 | +0.00082 ± 0.00039 |  |  |
| W16 + Pw | 0.3 | -0.00112 ± 0.00041 | -0.00008 ± 0.00038 | +0.00104 ± 0.00047 | -0.00027 ± 0.00022 | -0.00049 ± 0.00021 |
| W16 + Pw | 0.4 | -0.00141 ± 0.00047 | -0.00043 ± 0.00044 | +0.00099 ± 0.00052 | -0.00062 ± 0.00030 | -0.00078 ± 0.00030 |
| W16 + Px8 | 0.1 | -0.00028 ± 0.00022 | +0.00002 ± 0.00020 | +0.00031 ± 0.00029 | +0.00003 ± 0.00019 | +0.00051 ± 0.00024 |
| W16 + Px8 | 0.2 | -0.00079 ± 0.00033 | -0.00001 ± 0.00029 | +0.00078 ± 0.00040 |  |  |
| W16 + Px8 | 0.3 | -0.00109 ± 0.00041 | -0.00024 ± 0.00038 | +0.00085 ± 0.00047 | -0.00023 ± 0.00021 | -0.00030 ± 0.00020 |
| W16 + Px8 | 0.4 | -0.00139 ± 0.00048 | -0.00044 ± 0.00044 | +0.00095 ± 0.00053 | -0.00043 ± 0.00029 | -0.00059 ± 0.00030 |

| cross-composition (private-like view) | delta |
|---|---|
| [W16 + 0.2 Pd] - [W8 + 0.2 Pd = the TO half of W8P20] | -0.00014 ± 0.00024 |
| [W16 + 0.2 Px8] - [W8 + 0.2 Pd = the TO half of W8P20] | -0.00001 ± 0.00024 |
| [W16 + 0.3 Px8] - [W8 + 0.2 Pd = the TO half of W8P20] | -0.00024 ± 0.00027 |
| [W16 + 0.4 Px8] - [W8 + 0.2 Pd = the TO half of W8P20] | -0.00043 ± 0.00032 |
| [W16 + 0.4 Pw] - [W8 + 0.2 Pd = the TO half of W8P20] | -0.00042 ± 0.00034 |
| [W8 + 0.4 Px8] - [W8 + 0.2 Pd = the TO half of W8P20] | -0.00051 ± 0.00028 |
| W16 - W8 | +0.00024 ± 0.00027 (lag 0: +0.00006 ± 0.00027) |
| W16 - [W8 + 0.2 Pd] | +0.00000 ± 0.00034 |

### 3.5 Answer

| question | val FINAL | pmix-private | stale-val TO | lagged holdout TO (long feeds) | answer by the pre-declared rules |
|---|---|---|---|---|---|
| scored form W8 + Pd: 0.1 / 0.3 / 0.4 against 0.2 | -0.00019 / -0.00036 / -0.00019 | -0.00037 / -0.00039 / -0.00013 | -0.00051 / -0.00011 / +0.00033 | -0.00011 / -0.00039 / -0.00077 | **0.2 stands** (no weight passes R2 with or without the lagged clause) |
| W16 + Px8 (the lead's declared member set): 0.1 / 0.3 / 0.4 against 0.2 | -0.00008 / +0.00030 / +0.00055 ± 0.00031 | +0.00021 / +0.00056 / +0.00092 ± 0.00049 | -0.00029 / +0.00041 / +0.00082 ± 0.00033 | +0.00003 / -0.00023 / -0.00043 ± 0.00029 | 0.3 and 0.4 pass R2 without the lagged clause and FAIL with it: **not defensible as a blind change** |
| feed-size rule on W16 + Px8 (0.1 on M < 50, 0.4 on M >= 50), nested | +0.00065 (z 2.31) | +0.00113 | +0.00058 | 0.4 against 0.2 on long feeds -0.00043 ± 0.00029 | R3 "DEFENSIBLE" on val, block 3: **"val-only, contradicted on the lagged holdout"**; five compositions: mean nested +0.000298, "one composition only" by the letter |

1. **The scored weight 0.2 is the weight for any final that is sent without a public reading of its own.** For the scored form every witness agrees.
2. **Is there a defensible case for a heavier weight on private?** A case, yes; a defensible one by my rules, no. The case: on the 16-base with the wide popularity-light students the three val-sample witnesses prefer 0.4 on long feeds (same members, 0.4 against 0.2 on M >= 50 only: val FINAL +0.00064 ± 0.00021, pmix-private +0.00120 ± 0.00037, stale-val +0.00043 ± 0.00022; both val days, both user halves, 5 of 5 compositions in sign), with a mechanism (the two pools are level on the long feeds of val). Against it: the lagged holdout in 9 of 9 compositions (mean -0.00045 ± 0.00023) and the in-period holdout (-0.00111 ± 0.00022, reported). There is NO staleness bonus to lean on: public-like against private-like lag changes the admixture by +0.00006 ± 0.00018, and the label-free offsets on private days equal those on val days.
3. **A weight that depends on a label-free feed property?** The modal pick of the nested read is a feed-size rule in all five compositions; a cold-share rule is the second or third most frequent pick in two of them (17% and 8% of the half-samples) and never leads. What all five compositions agree on in sign is: long feeds tolerate (val: prefer) 0.4, short feeds are hurt by 0.4 (-0.0002 .. -0.0012 per short feed), 0.1 against 0.2 on short feeds is undecided. So a heavier weight, if ever adopted, belongs on long feeds only; a constant 0.3 - 0.4 gives away on the 39% short feeds part of what it gains.
4. My probability that 0.4 on long feeds beats 0.2 on private for a 16-base file with the wide popularity-light students: **0.55 - 0.6** (val alone would say 0.8; the lagged holdout alone 0.1 - 0.2; I weight val higher because it predicted the three public contrasts of this family and the lagged holdout reads the same family at zero where public reads +0.0005 .. +0.0007). Expected value of the change about +0.0002, with -0.0004 if the holdout family is right. That is a question for a public slot (section 5), not for a blind change of a final.

## 4. Assumption audit (extension of reports/STALE_1115/veto.md to claims that rest on val out-of-fold reads of the fine-tune, on arguments, or that the 11-15 public scores weaken)

### 4.1 The list

Types: (i) rests on the in-period or lagged holdout; (ii) rests on val out-of-fold reads of the fine-tune; (iii) an argument, not a measurement; (iv) contradicted or weakened by the 11-15 public scores. Status words: HOLDS, WEAKENED, NOT SUPPORTED, OPEN. Sources: CS = `reports/CURRENT_STATUS.md`, EK = `nn_x/w5/logs/E_kd_predeclared.txt`, SY = `reports/RECHECK_1114/SYNTHESIS.md`, FR = `reports/RECHECK_1114/ft_redesign.md`.

| # | standing claim (where) | type | what the data say [M unless marked] | status | what changes if it is wrong | cheapest test |
|---|---|---|---|---|---|---|
| 1 | "Val out-of-fold under-reads the fine-tune half because a val row is scored by 3 fold-models and a test row by 15" (CS section 1 bullet 3; EK 02:15Z (D')) | (iii) + (ii) | The fold noise of an FT pool on test rows is 0.26% of its within-feed variance (r between two fold pools 0.9974) and is common to all members (single member 0.31%, 16-pool 0.26%: the shared GroupKFold split is never averaged out by the pool). SIMEX on val: removing 4/5 of it is worth +0.00008 (structured noise) .. +0.00015 (white) at FINAL level. The gap to explain is +0.00048 (public) - +0.00009 (val) = 0.0004. | **NOT SUPPORTED as the main cause** (it carries about +0.0001). The measured alternative: the FT half is worth more on colder feeds: pmix-public +0.00066 ± 0.00054 (W8), +0.00053 (W16); pmix-private +0.00104 ± 0.00067 / +0.00071 ± 0.00064; label-free its cold promotion grows with the day (last table of section 2.2). | (a) More fold models or seeds will not recover the gap (see 4). (b) The FT half is a staleness adaptation and is worth MORE on private than on public: no final may drop or dilute it (TO-only twins, full-data twins whose FT half is weaker). (c) The lever with the right sign is the fine-tune under private-like stale inputs (STALE-FT, veto.md item 6): its prior goes up. | Done here (R5). Next: STALE-FT on 2 wide members judged out of fold on stale-val (3 GPU-h; on the lead's list as "prepared as code"). |
| 2 | "The fine-tune half adds 0 for students; expect the FT half of W16 to add 0.0000 ± 0.0004 on val" (FR verdict; SY) | (ii) | True on plain val (W8 +0.00009 ± 0.00043, W16 +0.00014 ± 0.00041). Not true at the test mix: public +0.00048 ± 0.00055 (W8), +0.00047 (G3); pmix-public / pmix-private as in row 1; half-stale (TO stale, FT fresh) +0.00149 ± 0.00043 (W8), +0.00090 ± 0.00043 (W16). | **WEAKENED** (the lead already keeps the FT half; the claim survives only for plain val) | New members must have their own fine-tune (hz re-cools, wide popularity-light students, full-data twins): the lead does this. | none needed |
| 3 | "w_ft = 0.5; w_ft != 0.5 is closed, nested pick worth 0 in every pool" (SY section 5; FR 85-97) | (ii) | R4: no w_ft > 0.5 reads >= +1.5 SE on pmix-private with val >= 0 in both W8 and W16new; w_ft 0.3-0.4 is +0.0003 on val for W8 only (not on W16new: +0.00002). Nested pick by pmix-private weights: -0.0001 .. -0.0005 against 0.5 in six pools. At the private mix w_ft 0 costs 0.0002 - 0.0010 in all six pools, w_ft 0.2 costs 0 - 0.0004, w_ft 0.7 - 0.8 reads between -0.0007 and +0.0004 with no consistent sign. | **HOLDS** (0.5 is a sound hedge; the val optimum below 0.5 does not carry to the private mix) | nothing | done (R4) |
| 4 | "More fine-tune seeds do not help" (registry line 72 says FT-seed bagging gives +0.0002; SY section 5 drops more seeds) | (ii) + public | Two scored pairs: 3 seeds against 1 on the same members: C2 - G3 public -0.00007 ± 0.00016 (val +0.00018 ± 0.00012); C3 - C1 public +0.00006 ± 0.00014 (val +0.00012 ± 0.00011). Row 1 explains why: seeds average seed noise, the fold noise is shared. | **HOLDS** (public: 0.0000 ± 0.0001; the val +0.00015 did not transfer) | nothing; "more folds / repeated K-fold with different splits" would be the only way to cut the fold noise and is bounded at about +0.0001 | none needed |
| 5 | "The pool is saturated at 16 members" (SY section 5 "size law saturated") | measurement on val + holdout | R6: a - b / n fits give b / 32 = +0.00006 .. +0.00008 on val FINAL, pmix-public, pmix-private, val TO, stale-val TO, val FT, holdout TO. 8 -> 16 measured +0.00003 .. +0.00027, fit +0.00012 .. +0.00016. | **HOLDS** on every witness including the stale ones | nothing | done (R6) |
| 6 | "W16 is better than W8 in expectation; W8 is a favourable val draw" (SY section 0 item 2; CS section 1) | (iii) + val | Public W16 - W8 = -0.000003 ± 0.00045. Forecast private W8 - W16 = +0.00004 ± 0.00036. The size law gives W16 +0.00014 in expectation; W8 - W8b on val +0.00026 ± 0.00048, on pmix-private +0.00054, on stale-val -0.00015. | **HOLDS as an argument, unmeasurable**: the two files are a coin flip on private | The role "pick 1 = W16" has no measured support against W8P20: see row 11 | none can resolve +0.00014 |
| 7 | "Narrow students hurt" (S16 public -0.00041; SY drops narrow students in the wide pool) | (iv) one public reading at 0.9 SE | val FINAL +0.00008 ± 0.00034, pmix-public -0.00022, pmix-private -0.00038 ± 0.00049, holdout -0.00028 ± 0.00019, stale-val TO +0.00010, SVF -0.00008. Forecast S16 - W16 on private -0.00039 ± 0.00040 (P(S16 better) 0.22). | **HOLDS weakly** (4 of 6 columns negative, none above 1.5 SE) | nothing | none needed |
| 8 | "Public-transfer coefficients are not identified: quote none" (CS log 21:30Z (f); SY section 5) | (iv) | With 6 independent scored pairs the slope of the public delta on the val FINAL delta is 1.08 ± 0.49, on pmix-public 0.78 ± 0.34; chi2 against slope 1: 2.1 and 1.8 on 6 df. Level: the 7 scored files sit within sd 0.00025 of the line on record (0.0006 on record). | **WEAKENED**: the slope is positive at 2.2 SE and compatible with 1; "a val FINAL delta transfers to public about one for one, within a factor 2" can now be quoted. The private slope cannot be checked; pmix-private reads 1.7 times the public deltas. | Rules that discount val deltas for transfer (thresholds in SE of val alone) are slightly conservative, not wrong | free: add each new scored pair to table T1f |
| 9 | "Soft-target fine-tune is closed" (SY section 5 row 1; FR section 8) | (ii) + low power | Closure = one CPU pilot: 2 members, 3 of 5 folds, 1 FT seed, plain val, file PROXY -0.00036 ± 0.00031 while the member gained +0.00053 ± 0.00038 [D FR 148-177]; the pilot's own statement: "effects above +0.0007 are excluded at about 2 SE, smaller ones are not". It had never been read at the private mix or on cold-heavy feeds, which is where row 1 puts the value of the fine-tune. I re-read the stored pilot arrays (block 6; control reproduces -0.00036 ± 0.00031 exactly; table T9 in 4.3): file proxy B - A under pmix-private weights -0.00044 ± 0.00046, on feeds with cold share >= 0.30 -0.00034 ± 0.00048; the member-level gain of the soft target is +0.00093 on cool feeds and +0.00002 on cold feeds, i.e. pulling the fine-tune towards the train-only pool undoes it exactly where it pays (the baseline fine-tune's file proxy is +0.00116 ± 0.00085 on cold feeds, +0.00007 on the others). | **HOLDS** (the closure also holds at the private mix; same low power: 2 members, 3 folds) | nothing | done (T9) |
| 10 | "No case for a heavier popularity-light weight" (CS log 02:19Z; from the lagged holdout of W8 + Pd at train-only level) | (i) lagged holdout | True for the scored form (W8 + Pd: 0.2 is the best constant on every FINAL column). On the 16-base with the wide popularity-light students val, pmix-private and stale-val prefer 0.4 on long feeds (5 of 5 compositions) and the nested feed-size rule reads +0.00045 .. +0.00065 on the three 16-base compositions. My lagged read (section 3.4): 0.4 against 0.2 is negative in 9 of 9 compositions (mean -0.00045 ± 0.00023; W16 + Px8 -0.00043 ± 0.00029). | **HOLDS by the pre-declared rules** (the lagged clause fails for every heavier weight); **contested** by all three val-sample witnesses on the 16-base (+0.00064 ± 0.00021 for 0.4 on long feeds) | the weight of every 16-base file (W16P20x8, FD16P20, X16P20): about +0.0005 if val is right, -0.0003 if the lagged holdout is | two public slots (section 5.2): paired public SE 0.00027 |
| 11 | "FINAL PAIR TODAY: W16 + W8P20; by rule A' W8P20 is a pick-2 file" (CS section 1; EK 02:15Z) | (iv) + rule | W8P20 - W16: public +0.00053 ± 0.00050, val +0.00056 ± 0.00038, pmix-private +0.00064, SVF +0.00110 ± 0.00038; forecast private +0.00071 ± 0.00040, P(W8P20 better) 0.92. Same-view control: the gain is specific to the view on two disjoint bases (section 2.3) and public rewarded the view three times (+0.00052, +0.00057, +0.00070). | **The order is not supported**: W8P20 is the file expected to score best; W16 is a hedge worth about +0.00002 in the expected maximum of the pair | If one file had to be named, naming W16 costs about 0.0007 in expectation | none needed; the +0.0008 threshold of A' asked one public reading to carry 2 SE alone, the combined evidence is now 3 scored contrasts + val + stale-val + lagged holdout |
| 12 | "W16P20x8 = the 16-student, 8-pop1 form of the scored W8P20 (no weight scan)" (EK 03:35Z) | (iii) | W16P20x8 - W8P20: val -0.00025 ± 0.00032, pmix-private -0.00041 ± 0.00048, stale-val TO -0.00011, SVF -0.00036; forecast -0.00034 ± 0.00049 (P(W16P20x8 better) 0.27). It passes its own build rule (val vs W16 +0.00031 >= 0). | **WEAKENED**: built as declared it is more likely below W8P20 than above it; the weight does not transfer from the 8-base with d128 members to the 16-base with wide members (sections 3.1-3.3) | a public slot spent on W16P20x8 as declared is expected to read -0.0003 ± 0.0004 against W8P20 | free: build the weight-0.4-on-long-feeds form next to it and choose which one to send (section 5) |
| 13 | "Private is staler and colder than public, so the popularity-light gain is larger there" (implicit in CS section 1: pmix-private +0.00079 against public +0.00052) | (iii) | Section 2.2: the direct lag manipulation gives +0.00006 ± 0.00018 between public-like and private-like lags; stale-val minus fresh val +0.00032 ± 0.00041 at TO level; label-free the admixture acts 14% less on private days than on public days; within val no monotone trend by cold share or day at FINAL level. | **NOT SUPPORTED beyond +0.0002**: expect private about equal to public for this contrast | forecasts that add a private bonus over-state; the pair choice does not change | none needed |
| 14 | "The in-period holdout stays a valid second witness for every non-staleness family" (CS section 1 EVIDENCE POLICY) | (iv) check | On the three scored pairs outside the staleness family the holdout has the public sign and size (W16 - W8 +0.00004 vs 0.00000; S16 - W8 -0.00028 vs -0.00041; W8 - C3 +0.00149 vs +0.00103), where plain val misses S16. | **HOLDS** on the evidence available (3 pairs) | nothing | none needed |
| 15 | "Level line public = 0.0074 + 0.963 val, residual sd 0.0006" (CS section 1) | measurement | Residuals of the 7 scored files: sd 0.00025, mean +0.00007; the three student files with a fine-tune half sit +0.0003 above the line, S16 / W8TO / C3 / G3 0.0001 - 0.0002 below. | **HOLDS** (tighter than quoted) | nothing | none needed |
| 16 | "Full-data twins are equivalent to their held-out originals (+0.00005 .. +0.0002)" (EK 04:52Z) | (iii) | Not testable from stored files. Relevant here: the twin FD16P20 has the form 0.8 z(16 wide) + 0.2 z(4 WIDE popularity-light). Its held-out original W16P20w reads against W8P20: val +0.00000 ± 0.00034, pmix-private -0.00002, forecast -0.00004 ± 0.00050 (a tie), against W16 +0.00072 ± 0.00053. The weight question of section 3 applies to it (RULEM2040w - W16P20w: val +0.00071 ± 0.00021, pmix-private +0.00093 ± 0.00034, stale-val +0.00033 ± 0.00022). | **OPEN** (argument; acceptance tests of fulldata.md are structural) | the twins inherit whatever weight is chosen | the public reading of the twin against W8P20 (planned by the lead) |
| 17 | "Lagged holdout: S_lag +0.00024 ± 0.00031, D +0.00074 ± 0.00040 (1.85 SE): the bias explanation holds" (CS section 1 and log 02:19Z; poplag_verify 0) | (i) lagged holdout, one composition | Nine compositions (3 bases x 3 popularity-light pools, section 3.4): S_lag(0.2) from -0.00014 to +0.00024, mean +0.00003 ± 0.00023; the record is the most favourable of the nine. D on the other 8 base seeds (W8b + Pd): +0.00015 ± 0.00039; on W16 + Pd +0.00032 ± 0.00041. The rule of 22:17Z (S_lag >= -0.0002) is met by all nine. | **WEAKENED in size, unchanged in outcome**: under the lag the in-period penalty (mean -0.00039) disappears; no gain appears. The lagged holdout reads at zero a family that public reads at +0.0005 .. +0.0007 (three contrasts) | "lagged holdout +" should not be counted as a positive witness next to val and public; the eligibility of the admixture does not change | done here (12 members scored, 33 CPU-min); for future reads use at least two base-seed sets |

Not re-audited here because veto.md covers them: checkpoint selection on the in-period holdout (veto 0 item 5), the 15 holdout closures of the staleness family, width 192, KD weight, cooldown mix.

### 4.2 Measurements behind the list

Fine-tune weight (rule R4):

**plain val**
| pool | w_ft 0.0 | w_ft 0.2 | w_ft 0.3 | w_ft 0.4 | w_ft 0.6 | w_ft 0.7 | w_ft 0.8 | w_ft 1.0 |
|---|---|---|---|---|---|---|---|---|
| W8 | -0.00009 ± 0.00043 | +0.00016 ± 0.00032 | +0.00028 ± 0.00026 | +0.00030 ± 0.00019 | -0.00007 ± 0.00017 | -0.00068 ± 0.00025 | -0.00065 ± 0.00031 | -0.00078 ± 0.00040 |
| W16new (s955-958) | -0.00017 ± 0.00038 | -0.00012 ± 0.00028 | +0.00002 ± 0.00024 | +0.00001 ± 0.00015 | +0.00005 ± 0.00019 | +0.00006 ± 0.00025 | -0.00029 ± 0.00031 | -0.00096 ± 0.00039 |
| W16 | -0.00014 ± 0.00041 | +0.00009 ± 0.00032 | -0.00009 ± 0.00026 | +0.00000 ± 0.00017 | -0.00019 ± 0.00016 | -0.00012 ± 0.00024 | -0.00029 ± 0.00029 | -0.00088 ± 0.00039 |
| W8P20 | -0.00002 ± 0.00042 | +0.00022 ± 0.00033 | +0.00005 ± 0.00027 | -0.00014 ± 0.00020 | -0.00024 ± 0.00016 | -0.00016 ± 0.00023 | -0.00028 ± 0.00029 | -0.00047 ± 0.00039 |
| W16P20x8 | -0.00016 ± 0.00042 | -0.00001 ± 0.00031 | -0.00011 ± 0.00025 | -0.00009 ± 0.00018 | +0.00014 ± 0.00017 | +0.00020 ± 0.00025 | -0.00014 ± 0.00031 | -0.00068 ± 0.00040 |
| S16 | -0.00001 ± 0.00041 | -0.00040 ± 0.00032 | -0.00018 ± 0.00026 | -0.00034 ± 0.00019 | -0.00022 ± 0.00017 | -0.00024 ± 0.00026 | -0.00037 ± 0.00031 | -0.00088 ± 0.00040 |

**pmix-public**
| pool | w_ft 0.0 | w_ft 0.2 | w_ft 0.3 | w_ft 0.4 | w_ft 0.6 | w_ft 0.7 | w_ft 0.8 | w_ft 1.0 |
|---|---|---|---|---|---|---|---|---|
| W8 | -0.00066 ± 0.00054 | -0.00018 ± 0.00040 | +0.00008 ± 0.00032 | +0.00019 ± 0.00023 | -0.00008 ± 0.00019 | -0.00066 ± 0.00030 | -0.00065 ± 0.00038 | -0.00072 ± 0.00050 |
| W16new (s955-958) | -0.00033 ± 0.00047 | -0.00030 ± 0.00033 | +0.00005 ± 0.00027 | +0.00001 ± 0.00017 | +0.00008 ± 0.00025 | +0.00010 ± 0.00032 | -0.00016 ± 0.00039 | -0.00095 ± 0.00050 |
| W16 | -0.00053 ± 0.00052 | -0.00017 ± 0.00041 | -0.00008 ± 0.00033 | -0.00004 ± 0.00021 | -0.00032 ± 0.00020 | -0.00021 ± 0.00030 | -0.00058 ± 0.00037 | -0.00099 ± 0.00049 |
| W8P20 | -0.00033 ± 0.00052 | +0.00009 ± 0.00042 | +0.00009 ± 0.00033 | -0.00024 ± 0.00024 | -0.00036 ± 0.00021 | -0.00032 ± 0.00029 | -0.00038 ± 0.00037 | -0.00060 ± 0.00049 |
| W16P20x8 | -0.00038 ± 0.00051 | -0.00010 ± 0.00038 | -0.00010 ± 0.00031 | +0.00001 ± 0.00023 | +0.00022 ± 0.00021 | +0.00034 ± 0.00030 | -0.00007 ± 0.00038 | -0.00068 ± 0.00050 |
| S16 | -0.00006 ± 0.00051 | -0.00014 ± 0.00039 | -0.00005 ± 0.00034 | -0.00006 ± 0.00025 | -0.00018 ± 0.00021 | +0.00007 ± 0.00032 | -0.00002 ± 0.00039 | -0.00040 ± 0.00050 |

**pmix-private**
| pool | w_ft 0.0 | w_ft 0.2 | w_ft 0.3 | w_ft 0.4 | w_ft 0.6 | w_ft 0.7 | w_ft 0.8 | w_ft 1.0 |
|---|---|---|---|---|---|---|---|---|
| W8 | -0.00104 ± 0.00067 | -0.00033 ± 0.00049 | -0.00000 ± 0.00039 | +0.00009 ± 0.00028 | -0.00006 ± 0.00023 | -0.00068 ± 0.00037 | -0.00057 ± 0.00046 | -0.00053 ± 0.00062 |
| W16new (s955-958) | -0.00045 ± 0.00056 | -0.00038 ± 0.00039 | +0.00003 ± 0.00032 | +0.00002 ± 0.00020 | +0.00014 ± 0.00031 | +0.00012 ± 0.00040 | -0.00007 ± 0.00048 | -0.00093 ± 0.00062 |
| W16 | -0.00071 ± 0.00064 | -0.00025 ± 0.00051 | +0.00002 ± 0.00040 | -0.00005 ± 0.00026 | -0.00034 ± 0.00024 | -0.00024 ± 0.00037 | -0.00065 ± 0.00045 | -0.00098 ± 0.00060 |
| W8P20 | -0.00059 ± 0.00064 | +0.00000 ± 0.00052 | +0.00013 ± 0.00040 | -0.00023 ± 0.00030 | -0.00050 ± 0.00026 | -0.00046 ± 0.00036 | -0.00045 ± 0.00046 | -0.00061 ± 0.00061 |
| W16P20x8 | -0.00043 ± 0.00062 | -0.00014 ± 0.00046 | -0.00005 ± 0.00038 | +0.00016 ± 0.00028 | +0.00032 ± 0.00025 | +0.00042 ± 0.00037 | +0.00001 ± 0.00047 | -0.00071 ± 0.00061 |
| S16 | -0.00015 ± 0.00064 | -0.00007 ± 0.00049 | +0.00000 ± 0.00043 | +0.00010 ± 0.00032 | -0.00018 ± 0.00025 | +0.00020 ± 0.00039 | +0.00019 ± 0.00049 | -0.00012 ± 0.00062 |

**val M>=50**
| pool | w_ft 0.0 | w_ft 0.2 | w_ft 0.3 | w_ft 0.4 | w_ft 0.6 | w_ft 0.7 | w_ft 0.8 | w_ft 1.0 |
|---|---|---|---|---|---|---|---|---|
| W8 | -0.00002 ± 0.00049 | +0.00031 ± 0.00036 | +0.00035 ± 0.00028 | +0.00052 ± 0.00019 | +0.00010 ± 0.00018 | -0.00055 ± 0.00028 | -0.00081 ± 0.00035 | -0.00071 ± 0.00046 |
| W16new (s955-958) | -0.00042 ± 0.00045 | +0.00010 ± 0.00033 | +0.00018 ± 0.00028 | -0.00005 ± 0.00017 | +0.00011 ± 0.00022 | +0.00023 ± 0.00029 | +0.00012 ± 0.00038 | -0.00038 ± 0.00046 |
| W16 | -0.00011 ± 0.00048 | +0.00041 ± 0.00039 | -0.00001 ± 0.00031 | +0.00005 ± 0.00021 | -0.00016 ± 0.00018 | +0.00008 ± 0.00027 | -0.00013 ± 0.00034 | -0.00066 ± 0.00045 |
| W8P20 | -0.00001 ± 0.00051 | +0.00031 ± 0.00039 | +0.00021 ± 0.00032 | -0.00006 ± 0.00023 | -0.00016 ± 0.00019 | -0.00032 ± 0.00027 | -0.00047 ± 0.00033 | -0.00046 ± 0.00042 |
| W16P20x8 | -0.00050 ± 0.00051 | -0.00017 ± 0.00038 | -0.00022 ± 0.00030 | -0.00004 ± 0.00024 | +0.00006 ± 0.00020 | -0.00000 ± 0.00029 | -0.00029 ± 0.00036 | -0.00064 ± 0.00046 |
| S16 | +0.00008 ± 0.00049 | +0.00004 ± 0.00038 | +0.00015 ± 0.00032 | -0.00021 ± 0.00021 | -0.00029 ± 0.00020 | -0.00019 ± 0.00031 | -0.00002 ± 0.00036 | -0.00029 ± 0.00046 |

**val M<50**
| pool | w_ft 0.0 | w_ft 0.2 | w_ft 0.3 | w_ft 0.4 | w_ft 0.6 | w_ft 0.7 | w_ft 0.8 | w_ft 1.0 |
|---|---|---|---|---|---|---|---|---|
| W8 | -0.00018 ± 0.00076 | -0.00006 ± 0.00059 | +0.00017 ± 0.00049 | -0.00001 ± 0.00036 | -0.00031 ± 0.00032 | -0.00087 ± 0.00044 | -0.00042 ± 0.00055 | -0.00088 ± 0.00073 |
| W16new (s955-958) | +0.00018 ± 0.00068 | -0.00045 ± 0.00049 | -0.00023 ± 0.00041 | +0.00011 ± 0.00027 | -0.00003 ± 0.00034 | -0.00018 ± 0.00043 | -0.00088 ± 0.00053 | -0.00181 ± 0.00070 |
| W16 | -0.00019 ± 0.00071 | -0.00036 ± 0.00055 | -0.00022 ± 0.00044 | -0.00007 ± 0.00030 | -0.00022 ± 0.00029 | -0.00041 ± 0.00044 | -0.00053 ± 0.00053 | -0.00120 ± 0.00071 |
| W8P20 | -0.00004 ± 0.00073 | +0.00009 ± 0.00059 | -0.00019 ± 0.00046 | -0.00024 ± 0.00035 | -0.00036 ± 0.00027 | +0.00008 ± 0.00039 | -0.00001 ± 0.00052 | -0.00048 ± 0.00072 |
| W16P20x8 | +0.00034 ± 0.00071 | +0.00022 ± 0.00053 | +0.00005 ± 0.00042 | -0.00017 ± 0.00028 | +0.00025 ± 0.00030 | +0.00050 ± 0.00044 | +0.00008 ± 0.00054 | -0.00073 ± 0.00072 |
| S16 | -0.00015 ± 0.00072 | -0.00104 ± 0.00054 | -0.00067 ± 0.00044 | -0.00053 ± 0.00034 | -0.00012 ± 0.00031 | -0.00032 ± 0.00045 | -0.00086 ± 0.00055 | -0.00173 ± 0.00072 |

**half-stale (TO branch from stale-val files; information)**
| pool | w_ft 0.0 | w_ft 0.2 | w_ft 0.3 | w_ft 0.4 | w_ft 0.6 | w_ft 0.7 | w_ft 0.8 | w_ft 1.0 |
|---|---|---|---|---|---|---|---|---|
| W8 | -0.00149 ± 0.00043 | -0.00044 ± 0.00034 | -0.00047 ± 0.00028 | -0.00019 ± 0.00019 | -0.00006 ± 0.00020 | -0.00012 ± 0.00029 | -0.00012 ± 0.00036 | -0.00043 ± 0.00046 |
| W16new (s955-958) | -0.00083 ± 0.00042 | -0.00085 ± 0.00032 | -0.00061 ± 0.00026 | -0.00026 ± 0.00018 | -0.00016 ± 0.00018 | +0.00005 ± 0.00027 | -0.00005 ± 0.00032 | -0.00037 ± 0.00043 |
| W16 | -0.00090 ± 0.00043 | -0.00047 ± 0.00032 | -0.00048 ± 0.00026 | -0.00021 ± 0.00018 | +0.00008 ± 0.00019 | +0.00009 ± 0.00027 | +0.00010 ± 0.00033 | -0.00014 ± 0.00045 |
| W8P20 | -0.00125 ± 0.00043 | -0.00027 ± 0.00033 | -0.00009 ± 0.00026 | +0.00014 ± 0.00019 | -0.00027 ± 0.00021 | -0.00046 ± 0.00028 | -0.00043 ± 0.00034 | -0.00028 ± 0.00045 |
| W16P20x8 | -0.00100 ± 0.00043 | -0.00016 ± 0.00033 | -0.00028 ± 0.00026 | -0.00022 ± 0.00018 | +0.00021 ± 0.00019 | +0.00022 ± 0.00027 | +0.00015 ± 0.00033 | -0.00038 ± 0.00043 |
| S16 | -0.00131 ± 0.00043 | -0.00050 ± 0.00034 | -0.00040 ± 0.00026 | -0.00012 ± 0.00019 | +0.00016 ± 0.00019 | +0.00020 ± 0.00027 | -0.00009 ± 0.00033 | -0.00038 ± 0.00044 |

**R4:** w_ft > 0.5 passing (pmix-private >= +1.5 SE and val >= 0 in BOTH W8 and W16new): none -> "w_ft = 0.5" is not weakened by this scan. w_ft < 0.5 with val >= +1.5 SE in both: none.

Nested pick of w_ft (grid above; picked on one user half by plain val or by pmix-private-weighted val, scored on the other; 200 splits, both directions; delta against 0.5):
| pool | pick by plain val: mean picked w_ft | out-of-sample val | out-of-sample pmix-private | pick by pmix-private weights: mean picked w_ft | out-of-sample val | out-of-sample pmix-private |
|---|---|---|---|---|---|---|
| W8 | 0.33 | +0.00001 (sd 0.00022) | -0.00031 (sd 0.00036) | 0.48 | -0.00007 (sd 0.00029) | -0.00041 (sd 0.00034) |
| W16new (s955-958) | 0.46 | -0.00025 (sd 0.00017) | -0.00026 (sd 0.00030) | 0.53 | -0.00018 (sd 0.00020) | -0.00033 (sd 0.00029) |
| W16 | 0.35 | -0.00026 (sd 0.00019) | -0.00051 (sd 0.00031) | 0.42 | -0.00021 (sd 0.00024) | -0.00046 (sd 0.00034) |
| W8P20 | 0.31 | -0.00013 (sd 0.00027) | -0.00034 (sd 0.00035) | 0.42 | -0.00016 (sd 0.00024) | -0.00046 (sd 0.00037) |
| W16P20x8 | 0.50 | -0.00017 (sd 0.00023) | -0.00009 (sd 0.00038) | 0.56 | -0.00006 (sd 0.00021) | -0.00009 (sd 0.00038) |
| S16 | 0.31 | -0.00031 (sd 0.00018) | -0.00031 (sd 0.00028) | 0.54 | -0.00052 (sd 0.00026) | -0.00046 (sd 0.00030) |

FT half value (FINAL w_ft 0.5 minus TO only) by feed cold share and by M, val:
| pool | cold < 0.15 | 0.15-0.30 | 0.30-0.45 | >= 0.45 | M < 50 | M >= 50 | Tue | Thu |
|---|---|---|---|---|---|---|---|---|
| W8 | +0.00044 ± 0.00148 | -0.00031 ± 0.00059 | -0.00007 ± 0.00072 | +0.00257 ± 0.00168 | +0.00018 ± 0.00076 | +0.00002 ± 0.00049 | +0.00005 ± 0.00057 | +0.00012 ± 0.00063 |
| W16new (s955-958) | +0.00105 ± 0.00134 | -0.00021 ± 0.00055 | +0.00029 ± 0.00065 | +0.00078 ± 0.00126 | -0.00018 ± 0.00068 | +0.00042 ± 0.00045 | +0.00020 ± 0.00053 | +0.00014 ± 0.00056 |
| W16 | +0.00005 ± 0.00149 | -0.00022 ± 0.00056 | +0.00041 ± 0.00070 | +0.00119 ± 0.00160 | +0.00019 ± 0.00071 | +0.00011 ± 0.00048 | -0.00002 ± 0.00055 | +0.00032 ± 0.00061 |
| W8P20 | -0.00030 ± 0.00151 | -0.00041 ± 0.00060 | +0.00034 ± 0.00070 | +0.00144 ± 0.00162 | +0.00004 ± 0.00073 | +0.00001 ± 0.00051 | -0.00014 ± 0.00058 | +0.00019 ± 0.00062 |
| W16P20x8 | -0.00055 ± 0.00163 | +0.00010 ± 0.00058 | +0.00050 ± 0.00071 | -0.00014 ± 0.00146 | -0.00034 ± 0.00071 | +0.00050 ± 0.00051 | +0.00009 ± 0.00059 | +0.00023 ± 0.00059 |
| S16 | +0.00084 ± 0.00162 | -0.00011 ± 0.00059 | -0.00019 ± 0.00066 | +0.00062 ± 0.00162 | +0.00015 ± 0.00072 | -0.00008 ± 0.00049 | -0.00010 ± 0.00056 | +0.00013 ± 0.00061 |
| feeds | 1370 | 6763 | 5077 | 1213 | 5893 | 8530 | 7542 | 6878 |

Fold-to-fold agreement of the fine-tune branch on test rows and the SIMEX read (rules R5 / R5b):

| unit | feed-mean Pearson r between two fold predictions (10 pairs) | v1 = 1 - r | r(fold k, 5-fold mean) | public feeds r / private feeds r |
|---|---|---|---|---|
| single member (mean of the 16 wide students) | 0.9969 | 0.0031 | 0.9989 | 0.9969 / 0.9969 |
| W8 FT pool (8 members) | 0.9973 | 0.0027 | 0.9989 (same top-1 item 0.927) | 0.9973 / 0.9974 |
| W16 FT pool (16 members) | 0.9974 | 0.0026 | 0.9990 (same top-1 item 0.928) | 0.9974 / 0.9974 |
| Pd FT pool (4 pop1 d128 students) | 0.9979 | 0.0021 | 0.9991 (same top-1 item 0.931) | 0.9978 / 0.9979 |

Fold noise of a single member v1 = 0.0031; of the 8-pool 0.0027; of the 16-pool 0.0026. If the fold noise were independent across members the 8-pool would have about 0.0004 and the 16-pool 0.0002: the part common to all members (same GroupKFold split) is what a val row keeps and a test row averages over 5 folds.

**T4c. SIMEX on val: what removing 4/5 of the fold noise of the out-of-fold FT pool would be worth (rule R5 white noise, R5b structured noise)**
| pool | noise | FINAL val at lambda 0 | lambda 0.5 | lambda 1 | lambda 2 | extrapolated to lambda -0.8 (quadratic) | same, linear fit | gain FINAL at -0.8 (quadratic / linear) | FT branch alone: gain at -0.8 (quadratic / linear) |
|---|---|---|---|---|---|---|---|---|---|
| W8 | white | 0.40165 | 0.40156 | 0.40159 | 0.40159 | 0.40177 | 0.40163 | **+0.00012** / -0.00002 | +0.00044 / +0.00014 |
| W16 | white | 0.40158 | 0.40156 | 0.40144 | 0.40139 | 0.40172 | 0.40166 | **+0.00015** / +0.00008 | -0.00030 / +0.00048 |
| W16 | struct | 0.40158 | 0.40152 | 0.40152 | 0.40148 | 0.40166 | 0.40160 | **+0.00008** / +0.00002 | -0.00005 / +0.00015 |

**R5:** structured-noise SIMEX gain at FINAL level for W16 = +0.00008 (white noise: W8 +0.00012, W16 +0.00015). Bars: SUPPORTED iff >= 0.0002; NOT SUPPORTED AS THE MAIN CAUSE iff < 0.0001. -> NOT SUPPORTED AS THE MAIN CAUSE. For comparison: public reads the FT half of W8 at +0.00048 ± 0.00055, plain val at +0.00009 ± 0.00043, pmix-public at +0.00066 ± 0.00054.

The SIMEX extrapolations are noisy (20-30 noise draws; the lambda points differ by 0.0001 or less), so the honest statement is an order of magnitude: doubling the fold noise costs about 0.0001 at FINAL level, removing four fifths of it gains about +0.0001 (my three quadratic estimates +0.00008 .. +0.00015; linear fits -0.00002 .. +0.00008).

Pool size (rule R6):

| column | n = 1 | n = 2 | n = 4 | n = 8 | n = 16 | a (n = infinity) | b | expected gain of 16 -> 32 (b / 32) | 8 -> 16 measured | 8 -> 16 by the fit (b / 16) |
|---|---|---|---|---|---|---|---|---|---|---|
| val FINAL | 0.39957 (0.00040) | 0.40064 (0.00044) | 0.40133 (0.00036) | 0.40146 (0.00025) | 0.40158 (0.00000) | 0.40176 | 0.00219 | +0.00007 | +0.00012 | +0.00014 |
| pmix-public FINAL | 0.39634 (0.00046) | 0.39752 (0.00053) | 0.39824 (0.00038) | 0.39848 (0.00032) | 0.39874 (0.00000) | 0.39884 | 0.00253 | +0.00008 | +0.00026 | +0.00016 |
| pmix-private FINAL | 0.40188 (0.00068) | 0.40310 (0.00066) | 0.40378 (0.00046) | 0.40403 (0.00039) | 0.40429 (0.00000) | 0.40440 | 0.00253 | +0.00008 | +0.00025 | +0.00016 |
| val TO | 0.39927 (0.00047) | 0.40043 (0.00042) | 0.40108 (0.00041) | 0.40137 (0.00024) | 0.40144 (0.00000) | 0.40163 | 0.00236 | +0.00007 | +0.00007 | +0.00015 |
| stale-val TO | 0.39788 (0.00043) | 0.39901 (0.00043) | 0.39953 (0.00036) | 0.39967 (0.00025) | 0.39994 (0.00000) | 0.40003 | 0.00213 | +0.00007 | +0.00027 | +0.00013 |
| val FT | 0.39893 (0.00041) | 0.39994 (0.00046) | 0.40055 (0.00034) | 0.40067 (0.00025) | 0.40070 (0.00000) | 0.40092 | 0.00196 | +0.00006 | +0.00003 | +0.00012 |
| holdout TO | 0.32255 (0.00023) | 0.32355 (0.00018) | 0.32405 (0.00018) | 0.32436 (0.00013) | 0.32460 (0.00000) | 0.32464 | 0.00213 | +0.00007 | +0.00024 | +0.00013 |

**R6:** largest b / 32 over the columns = +0.00008 -> "saturated at 16" is CONFIRMED (< 0.0002 on every column).

### 4.3 More fine-tune seeds, the level line, transfer slopes, the soft-target pilot at the private mix

**T5a. More fine-tune seeds (3 seeds per fold against 1), scored pairs on record**
| pair | val FINAL | pmix-public | pmix-private | PUBLIC ± SE |
|---|---|---|---|---|
| C2 - G3 (G3 members, FT 3 seeds - 1 seed) | +0.00018 ± 0.00012 | +0.00023 ± 0.00015 | +0.00021 ± 0.00019 | -0.000066 ± 0.00016 |
| C3 - C1 (G3 + pop1 x8, FT 3 seeds - 1 seed) | +0.00012 ± 0.00011 | +0.00018 ± 0.00013 | +0.00022 ± 0.00016 | +0.000060 ± 0.00014 |
| C1 - G3 (pop1 x8 added, FT 1 seed) | +0.00057 ± 0.00040 | +0.00090 ± 0.00050 | +0.00107 ± 0.00062 | +0.000574 ± 0.00051 |
| C3 - C2 (pop1 x8 added, FT 3 seeds) | +0.00051 ± 0.00040 | +0.00085 ± 0.00051 | +0.00108 ± 0.00063 | +0.000700 ± 0.00052 |

**T5b. Level: the 7 scored files against the line on record (public = 0.0074 + 0.963 val; no refit)**
| file | val FINAL | PUBLIC | line | residual | pmix-public level | PUBLIC - pmix-public level |
|---|---|---|---|---|---|---|
| W8P20 | 0.40214 | 0.395009 | 0.39466 | +0.00035 | 0.39923 | -0.00422 |
| W8 | 0.40165 | 0.394486 | 0.39419 | +0.00030 | 0.39860 | -0.00411 |
| W16 | 0.40158 | 0.394483 | 0.39412 | +0.00036 | 0.39874 | -0.00426 |
| S16 | 0.40173 | 0.394078 | 0.39426 | -0.00019 | 0.39838 | -0.00430 |
| W8TO | 0.40157 | 0.394009 | 0.39411 | -0.00010 | 0.39794 | -0.00393 |
| C3 | 0.40101 | 0.393454 | 0.39357 | -0.00012 | 0.39812 | -0.00467 |
| G3 | 0.40032 | 0.392820 | 0.39291 | -0.00009 | 0.39705 | -0.00423 |
Residual sd over the 7 files 0.00025 (on record: 0.0006); mean +0.00007.

**T5c. Slope of the public delta on each val column, 6 independent scored pairs (through the origin; SE from the public SEs only)**
| column | slope ± SE | chi2 of public against slope = 1 (6 df) |
|---|---|---|
| val FINAL | +1.08 ± 0.49 | 2.1 |
| pmix-public FINAL | +0.78 ± 0.34 | 1.8 |
| pmix-private FINAL | +0.59 ± 0.26 | 4.1 |
| SVF | +0.56 ± 0.27 | 5.1 |

**T9. Soft-target fine-tune pilot re-read (8655 feeds done by all arms of both members; final EMA; mean of the two members; paired per feed)**
| quantity | plain val (pilot: see right) | pmix-private weights | cold share >= 0.30 | cold share < 0.30 | M >= 50 | M < 50 |
|---|---|---|---|---|---|---|
| file proxy, B soft targets - A baseline | -0.00036 ± 0.00031 (pilot -0.00036 ± 0.00031) | -0.00044 ± 0.00046 | -0.00034 ± 0.00048 | -0.00037 ± 0.00040 | -0.00002 ± 0.00033 | -0.00084 ± 0.00058 |
| member - TO, B - A | +0.00053 ± 0.00038 (pilot +0.00053 ± 0.00038) | +0.00012 ± 0.00056 | +0.00002 ± 0.00058 | +0.00093 ± 0.00051 | +0.00047 ± 0.00047 | +0.00062 ± 0.00065 |
| file proxy of the baseline A - TO pool | +0.00054 ± 0.00053 (pilot +0.00054 ± 0.00053) | +0.00143 ± 0.00086 | +0.00116 ± 0.00085 | +0.00007 ± 0.00068 | -0.00018 ± 0.00063 | +0.00157 ± 0.00094 |
CONTROL (plain val within 0.00003 of -0.00036): VALID. RULE (block 6): pmix-private -0.96 SE, cold >= 0.30 -0.71 SE -> the closure also holds at the private mix (same low power).

**T10. Popularity-light weight per branch: FINAL = 0.5 z((1 - a) z(base FT) + a z(pop FT)) + 0.5 z((1 - b) z(base TO) + b z(pop TO)); delta against the base (a = b = 0)**

**W8 + Pd** (cells: val FINAL ± SE / pmix-private / half-stale)
| weight in the FT branch (a) | TO branch b = 0 | b = 0.2 | b = 0.4 |
|---|---|---|---|
| a = 0.0 | 0 | +0.00024 ± 0.00024 / +0.00020 / +0.00034 | +0.00071 ± 0.00034 / +0.00117 / +0.00065 |
| a = 0.2 | +0.00038 ± 0.00022 / +0.00028 / +0.00029 | +0.00049 ± 0.00031 / +0.00079 / +0.00064 | +0.00052 ± 0.00040 / +0.00093 / +0.00080 |
| a = 0.4 | +0.00035 ± 0.00031 / +0.00048 / +0.00053 | +0.00026 ± 0.00039 / +0.00050 / +0.00058 | +0.00030 ± 0.00047 / +0.00066 / +0.00070 |

**W16 + Px8** (cells: val FINAL ± SE / pmix-private / half-stale)
| weight in the FT branch (a) | TO branch b = 0 | b = 0.2 | b = 0.4 |
|---|---|---|---|
| a = 0.0 | 0 | +0.00035 ± 0.00023 / +0.00064 / +0.00076 | +0.00054 ± 0.00034 / +0.00061 / +0.00133 |
| a = 0.2 | +0.00027 ± 0.00021 / +0.00043 / +0.00024 | +0.00031 ± 0.00031 / +0.00023 / +0.00075 | +0.00072 ± 0.00039 / +0.00074 / +0.00094 |
| a = 0.4 | +0.00032 ± 0.00030 / +0.00046 / +0.00050 | +0.00041 ± 0.00037 / +0.00055 / +0.00099 | +0.00086 ± 0.00044 / +0.00115 / +0.00133 |

Reading of T10 [M]: no cell differs from its neighbours by more than 1 SE; the admixture placed in the train-only branch alone (a = 0, b = 0.2) reads +0.00024 / +0.00035 against +0.00049 / +0.00031 in both branches, so "new members do not need their own fine-tune" (ft_redesign) is not contradicted on val; nothing here supports a branch-specific weight.

## 5. Recommendation

### 5.1 The final pair I would pick today

**W8P20 and W16: the lead's two files, with the roles reversed.** Only files with a public reading qualify today, and among them W8P20 is the one expected to score best on private (section 2.1). W16 is the hedge against "the popularity-light view reverses on private" (P about 0.08 by the rule, 0.15 as my own number); as a hedge it is worth about +0.00002 in the expected maximum of the pair [I: E(max(0, X)) for X ~ N(-0.00071, 0.00051)]. If ONE file had to be named, it is W8P20.

If the better of two finals counts (owner's open question 1), the second slot is worth more as a second popularity-light file of a different composition than as this hedge: a file that is a forecast tie with W8P20 at a realised sd of 0.0005 adds about +0.0002 to the expected maximum, ten times the hedge, at the price of both finals leaning on the same view. The candidates for that role are the full-data twin of the lead's plan (FD16P20; its held-out original W16P20w reads a tie with W8P20: -0.00004 ± 0.00050) and the outcome of the experiment below; each needs its own public reading first. Risk preference is the owner's: with W16 as the second final the pair cannot lose more than the hedge case; with two popularity-light finals it gains about +0.0002 in expectation and loses about 0.0007 against W16 in the 8 - 15% case.

What I would NOT do: (1) send W16P20x8 as declared in the expectation that it beats W8P20 (forecast -0.00034 ± 0.00049); (2) change the popularity-light weight of any final away from 0.2 on the val evidence alone (the lagged holdout contradicts every heavier weight, section 3.4); (3) add a private "staleness bonus" to any forecast of this family (section 2.2 (c)); (4) drop or dilute the fine-tune half of any final (audit rows 1-2: it is worth more on private than on val).

### 5.2 The single experiment that would most change the choice

**Two public slots for the same 24 members at two weights: W16P20x8 (as declared by the lead at 03:35Z) and RULEM2040 = per branch (1 - w) z(W16) + w z(Px8) with w = 0.2 on feeds with M < 50 and w = 0.4 on feeds with M >= 50, FINAL 0.5 / 0.5.** CPU build from members that are all fine-tuned already (the per-feed weight is applied to per-feed z, as in `t3b_rule.py`; the builder of the lead needs one `np.where` on the feed size); gate as usual.

Why this one.
- It is the one open question on which the witnesses disagree in sign at 1.5 - 5 SE each: val FINAL +0.00064 ± 0.00021, pmix-public +0.00091 ± 0.00029, pmix-private +0.00120 ± 0.00037, stale-val +0.00043 ± 0.00022 for RULEM2040 against W16P20x8; lagged holdout -0.00043 ± 0.00029 and in-period holdout -0.00111 ± 0.00022 per long feed. It is the val-against-holdout conflict that public settled once for this family, now between val and the LAGGED holdout, which is the witness the new evidence policy leans on.
- The two files differ only in the weight on long feeds, so the difference of their public scores has a paired SE of 0.00027 (a usual pair: 0.0004 - 0.0005) [M: sd of the val per-feed delta / sqrt(8,562)]. Val predicts about +0.0009 (in-sample; nested about +0.0005), the lagged holdout about -0.0003, the in-period holdout about -0.0007: the two families are 3 - 6 public SE apart. No other pair on the table separates them that well.
- Label-free the two files differ on private days as they do on public days (same top-1 item in 95.3% / 95.7% of feeds, cold-minus-warm offset +0.0046 / +0.0040, val days +0.0041 / +0.0043; `out/t2_forecast.md` T2h, last pair but one): what public reads for this pair is what private will pay, about 13% weaker.
- It decides something that every later final inherits: the weight of FD16P20, of X16P20 on an hz / xd / th base, and whether val + stale-val (RULE hz) or the lagged holdout should be believed when they disagree.

With ONE slot only: RULEM2040 alone, read against W8P20 (public SE of the pair 0.00045; pmix-public expectation +0.00064 in-sample) and against W16. It is less clean (different base, different popularity-light members) but it is the file with the higher upside of the two.

Reading rules I would write before the slot (the lead's own forms; thresholds in public SE):
- weight: "0.4 on long feeds is adopted for the 16-base finals iff public(RULEM2040) - public(W16P20x8) >= +0.0003 (1 paired SE) ; it is closed iff <= -0.0003; in between the weight stays 0.2";
- file: "RULEM2040 is a final candidate iff public >= W8P20 - 0.0003, and replaces W8P20 as the best file iff the inverse-variance mean of its val delta (+0.00039 ± 0.00035) and its public delta against W8P20 is >= +2 SE" (the form of the "ARM hz" file rule of 02:15Z; it needs public >= about +0.0008 against W8P20).

Why RULEM2040 and not the nested modal rule RULEM1040 (0.1 / 0.4): 0.1 against 0.2 on short feeds is mixed in sign over the five compositions and the lagged holdout cannot see short feeds at all; 0.4 against 0.2 on long feeds is the part on which 5 of 5 compositions agree on val and 9 of 9 disagree on the lagged holdout. RULEM2040 changes that one thing and keeps the scored weight elsewhere.

### 5.3 Next in line

1. STALE-FT (fine-tune on val with private-like stale inputs; veto.md item 6, "prepared as code"): the audit moves it up. The fine-tune half is the staleness adaptation of the file (worth +0.0005 on public, +0.0007 .. +0.0010 at the private mix, zero on cool feeds), and it is the only stage that trains on out-of-period labels; nothing so far trains it on inputs as stale as private rows. Judge: out-of-fold on stale-val, FINAL level.
2. For every read of the lagged holdout on this family: use more than one base-seed set (the D on record, +0.00074, is +0.00015 on the other 8 seeds) and quote the nine-composition mean (+0.00003 ± 0.00023 at weight 0.2).
3. Add each new scored pair to the sanity table T1f (free); six pairs give a transfer slope of 1.08 ± 0.49, ten would give about ± 0.35.

## 6. Method

- Loaders and compositions: `pm_common.py`, `t1_build.py` (member z = `stack_val.feed_rank_z[:, 1]` re-implemented in numba; pool = member mean of z; branch mix = (1 - w) z(base) + w z(pop); FINAL = 0.5 z(FT) + 0.5 z(TO), as `nn_x/w12/kd/build_r3.py`). Per-feed NDCG arrays of every composition and view are stored in `out/pf.npz` (rule files in `out/pf_rule.npz`), test and val FINAL scores in `cache/{testfin,valfin}_<name>.npy`.
- Feed-dependent rules are composed per feed from the constant-weight arrays (the per-feed z makes a per-feed weight well defined); a test-side rule file would be built the same way from the two constant-weight test scores.
- Nested reads: 200 random user-half splits (user-0 feeds are singletons), both directions, cross-fitted per-feed deltas over all 14,423 feeds; SE = median per-split paired SE.
- Forecast rule R1, weight rules R2 / R3, fine-tune weight R4, fold noise R5 / R5b, pool size R6, sanity R7: `PREDECLARED.txt` block 1 (before any number); the later blocks are dated additions, each before its read, each listing what had been seen.
- Fold statistics: `n2cpv3s3_test_folds.npy` holds, per fold, the mean over the 3 fine-tune seeds of the per-feed z on test rows (`nn_x/w5/finetune.py` line 280); GroupKFold without shuffle gives every member the same user folds.
- SIMEX: noise of variance lambda x v1 x var(pool) is added to the val out-of-fold FT pool (white: independent per row; structured: (P_A - P_B) / 2 of a random 8 / 8 split of the 16 students, which is uncorrelated with the 16-pool), FINAL NDCG at lambda 0.5 / 1 / 2, quadratic and linear extrapolation to lambda = -0.8.
- Lagged holdout: the witness agent's `lagged_score.py` imported read-only by `lag_run.py` with its output path redirected to `cache/lag/`; control = one member re-scored against the witness cache (max |logit| difference 1.53e-05); read-out `t6_lag_read.py`.
- Label-free day statistics (`t2_forecast.py`, T2h): per-feed Pearson r of the two FINAL z vectors, same top-1 item, cold-minus-warm offset (cold = `it_hist_all_luo` = 0), by Moscow day and public / private.

## 7. Limits, and what I did NOT check

Limits.
1. No private label exists. Every labelled number is from 14,423 val feeds of two weekdays (Tue, Thu) or from the train holdout. The "forecast" is rule R1, a combination I wrote before reading the new compositions but after reading the six scored pairs; its three val views share one sample, so their agreement is not three confirmations. The only independent sample is public (8,562 feeds, weekend).
2. pmix-private leans on few feeds: n_eff 6,389; 32.5% of the private weight sits on the 1,213 val feeds with cold share >= 0.45 (largest feed weight 5.95). It over-reads the six public pairs by a factor 1.7 (slope 0.59 ± 0.26).
3. SVF makes only the train-only branch stale. No fine-tuned prediction exists under stale inputs, so every FINAL-level statement about private assumes the fine-tune half behaves on private rows as it does out of fold on val. The one drift I see label-free is exactly in that half (its cold promotion on private is 1.4 - 2 times its val value).
4. Stale-val files are the witness / veto agents' CPU re-predictions; I checked them only through the reproduction of W8P20 - W8 (+0.00088 ± 0.00036) and used the veto agent's files for Pw on the strength of its statement that they equal the witness tool to 1.5e-5.
5. Seed draw: Pd and Pw are 4 members each, W8 / W8b 8 each. Differences between popularity-light compositions at the same weight (W8P20, its W8b twin, W16P20d / x8 / w) are within 1.5 SE and partly seed draw: the 16-base file reads 0.0004 below both of its 8-base twins at weight 0.2 (2 SE) and 0.0002 above them at 0.4.
6. Multiplicity (all looks): about 30 compositions x 3 references in the witness table; R2: 12 declared looks + 4; R3: 75 rules x 5 compositions (2 declared in block 1, 2 more run in the same pass without a prior declaration, 1 declared in block 4); 4 nested-pick lists; R4: 8 weights x 6 pools x 6 views; 10 slices x 12 pairs x 3 views in T7; the post hoc rows RULEM2040 / RULEM2040w (block 5); the pilot re-read (block 6); the per-branch grid (block 7, 2 x 9 cells). The first witness table was read once with 3 of 4 wide fine-tunes before the 4-of-4 re-run; R2-R6 were read at 4 of 4 only.
7. The lagged holdout keeps in-period labels, has long feeds only (M >= 50), is read at train-only level and on the 13,424 feeds where its label-free gate holds; I scored the private-like view only (no public-like view for the 12 members I added).
8. The feed-size rule was prompted by a table read on the full val sample (witness.md 3.4, W8 + Pd), then judged by a nested procedure on other compositions that share members with it: the nested z is optimistic by an amount I cannot compute.

What I did NOT check.
- No test-side build, no presub gate, no file: every new composition is an in-memory array (`cache/testfin_*.npy` are flat score vectors, not submissions).
- No GPU run of any kind; nothing on the hz / xd / th re-cools (not finished), on the full-data twins, or on @last checkpoints.
- Different popularity-light weights in the two branches: only the 3 x 3 information grid T10 (nothing above 1 SE); no nested read.
- Weights outside 0.1 - 0.4, three-group rules, rules on properties other than M and cold share (history length, user 0, day).
- A forecast of the private LEVEL of any file (only deltas).
- Whether the public / private split boundary and the day structure hide something val cannot show (private is 76% weekdays like val, public 0%): the popularity-light contrast is +0.0005 on both weekday val and weekend public, which is all I have.
- The leak rules for any new input: none was built here; no val or test label entered a model.

## 8. Commands

```
cd <repo>/nn_x/w12/stale/premortem
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1
PY=<repo>/.venv/bin/python
$PY t1_build.py                      # pools, compositions, per-feed arrays, test FINAL scores (16 s)
$PY t1_table.py                      # out/t1_table.md
$PY t3_weight.py                     # out/t3_weight.md (R2, R3)
$PY t3b_rule.py                      # out/pf_rule.npz (rule files in memory)
$PY t2_forecast.py out/pf_rule.npz   # out/t2_forecast.md (R1, nested pick, twins, gradient, day tables)
$PY t4_audit.py wft; $PY t4_audit.py fold; $PY t4_audit.py size     # out/t4_*.md (R4, R5, R6)
$PY t5_misc.py; $PY t7_robust.py; $PY t8_sameview.py; $PY t9_softft.py; $PY t10_branch.py
WT_THREADS=12 $PY lag_run.py control                                 # control of the lagged tool
WT_THREADS=12 $PY lag_run.py abl_e10_pop1kd05w192_s951,abl_e10_pop1kd05w192_s952,abl_e10_pop1kd05w192_s953,abl_e10_pop1kd05w192_s954 lag0,priv
WT_THREADS=12 $PY lag_run.py w5_e10_kd05w192_s955,w5_e10_kd05t25w192_s955,w5_e10_kd05w192_s956,w5_e10_kd05t25w192_s956,w5_e10_kd05w192_s957,w5_e10_kd05t25w192_s957,w5_e10_kd05w192_s958,w5_e10_kd05t25w192_s958 lag0,priv
$PY t6_lag_read.py                   # out/t6_lag.md
$PY make_report.py                   # this file (report_head.txt + blk_*.txt + out/*.md)
```

## 9. Files (all under `nn_x/w12/stale/premortem/` unless stated)

- `reports/STALE_1115/premortem.md` (this report); `PROGRESS.md`; `PREDECLARED.txt` (dated blocks).
- Code: `pm_common.py`, `t1_build.py`, `t1_table.py`, `t2_forecast.py`, `t3_weight.py`, `t3b_rule.py`, `t4_audit.py`, `t5_misc.py`, `t6_lag_read.py`, `t7_robust.py`, `t8_sameview.py`, `t9_softft.py`, `t10_branch.py`, `lag_run.py`, `make_report.py`, `report_head.txt`, `blk_*.txt`.
- Tables: `out/t1_table.md` (+ `_3of4`), `out/t2_forecast.md` / `.json`, `out/t3_weight.md` (+ `_run1`), `out/t4_wft.md`, `out/t4_fold.md`, `out/t4_size.md`, `out/t5_misc.md`, `out/t6_lag.md`, `out/t7_robust.md`, `out/t8_sameview.md`, `out/t9_softft.md`, `out/t10_branch.md`, `out/t1_pairs.json`, `out/avail.json`.
- Arrays: `out/pf.npz`, `out/pf_rule.npz` (per-feed NDCG), `cache/pool_*.npy` (pool z per kind), `cache/{testfin,valfin}_*.npy` (FINAL scores of compositions aligned to data/flat; NOT submission files), `cache/lag/{lag0,priv}/*.npy` (my lagged predictions), `cache/lag/control/`.
- Logs: `logs/`.

## 10. Compliance

CPU only (`CUDA_VISIBLE_DEVICES=""` in every command), 12 threads per process, two long processes of mine at most (the two `lag_run.py` jobs, 05:13-05:52Z). SELF-REPORTED: while those two ran I also started short table scripts (`t3_weight.py`, `t2_forecast.py`, `t7_robust.py`, `t8_sameview.py`, `t9_softft.py`, `t10_branch.py`, two one-off read-outs; 5 - 50 s each, 12 threads, one at a time); if a script of under a minute counts as heavy, the cap of two was exceeded for about 4 minutes in total. No process was stopped, signalled or reniced by me (mine ended by themselves); no pattern kill was used (`ps` / `pgrep` were used once each to LIST my own PIDs). No file outside `nn_x/w12/stale/premortem/` and `reports/STALE_1115/premortem.md` was written: the witness agent's tool was imported read-only with `PYTHONDONTWRITEBYTECODE=1`, its output path was redirected to my cache, my driver asserts beforehand that every raw column the tool would cache already exists in the witness directory, and my numba cache is in my directory. No install, no network, no submission file, nothing sent. Val labels were used only to score stored out-of-fold and train-only predictions; holdout labels only for train-only reads; no test label exists. Rules were written before the reads they judge (seven dated blocks), with one lapse that is listed in section 7 item 6: two of the five compositions of the R3 read (W8 + Px8, W16 + Pd) were run in the first pass without having been named in block 1. The report file itself is assembled by `make_report.py` from text templates (the session's write tool does not write report files directly; the task requires this file).
