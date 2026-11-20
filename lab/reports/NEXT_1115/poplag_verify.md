# poplag_verify: adversarial review of "poplag" (lagged-popularity holdout test) + the pre-declared SUBSET READ

Written 2026-11-15 by the reviewer. CPU only. Work directory `nn_x/w12/next/poplag_verify/`. Conventions: ± = 1 SE paired per feed, feed count given with each number; [M] = measured by me with my own code, [I] = inferred. "Subset" = holdout feeds with day >= Sep 13 (13,424 feeds, 2,089,648 rows); "complement" = the other 16,576 feeds. S(w) = NDCG((1 - w) z(W8 pool) + w z(pop1kd pool)) - NDCG(z(W8 pool)); S without a weight is w = 0.2.

## 0. Verdict

1. **SUBSET READ (rule of 2026-11-15T01:08:21Z), private-like, 13,424 feeds [M]: S_lag = +0.00024 ± 0.00031; S_0 = -0.00050 ± 0.00032; D = +0.00074 ± 0.00040 (1.85 SE). Branch: "bias explanation holds"** (pop-light admixture eligible as pick 2; as pick 1 only together with A' >= +0.0008).
2. Preconditions met with my own code [M]: G1-G5 hold on the subset (largest bucket gap 0.0496 / 0.0447 / 0.0492, bar 0.10); all 12 staged members confirmed on 400 subset feeds whose nine lagged columns I rebuilt and re-scored (max difference 1.5e-5).
3. **The study's verdict is CONFIRMED**: no read on the 30,000 feeds (private-like gate fails: `xp_imp_c50`, listings aged 10-24 d, gap 0.1161 > 0.10); control valid (-0.00047 ± 0.00023 against -0.00063 ± 0.00022); every number I re-derived agrees (section 1.2).
4. Strength: moderate. The penalty disappears, it does not become a gain (S_lag is 0.8 SE from zero). The rule returns "holds" in 4-8% of cases when the lag changes nothing [I]. The public-like view reads the same on the subset (S_lag +0.00018 ± 0.00031, D +0.00068 ± 0.00039) [M].
5. Main issue of the study: its "lagged" view matches test rows in coverage, not in time. 43% of the counted (row, dense day) pairs lie AFTER the row on the 30,000 feeds, 81% on the complement, 10% on the subset, 0% on Sep 17-19 [M]. Days >= Sep 13 are the right subset, for a stronger reason than the failed criterion.
6. Post-hoc [M]: inside the subset D sits on Sep 13-14 (5,145 feeds, +0.00154 ± 0.00072); Sep 17-19 (8,279 feeds, the causal part) gives +0.00024 ± 0.00046 from an S_0 already near zero (-0.00017 ± 0.00039). The gain is specific to the pop-light view: against a same-view admixture it changes by +0.00066 ± 0.00031 under the lag.

## 1. Measurements

### 1.1 The SUBSET READ (decision) [M]

Gate re-check, private-like view, subset rows against 1,829,811 private test rows of 12,186 feeds with M >= 50 (`v_subgate.py`; `it_hist_*` and `seq_cv_*` from the study's files, which I confirmed by my own rebuild on samples; `xp_imp_c50` is my own full rebuild):

| criterion | bar | `it_hist_all_luo` | `seq_cv_pop` | `xp_imp_c50` | result |
|---|---|---|---|---|---|
| G1 dZ (age-weighted zero-share gap) | <= 0.05 | 0.0156 | 0.0145 | 0.0225 | holds |
| G2 largest bucket with >= 5% of rows | <= 0.10 | 0.0496 (10-24 d) | 0.0447 (10-24 d) | 0.0492 (10-24 d) | holds |
| G3 dM (mean gap in test SD) | <= 0.20 | 0.0452 | 0.0377 | 0.0534 | holds |
| G4 `it_hist_all_luo` dZ <= one third of lag 0 | <= 0.0703 | 0.0156 (lag 0: 0.2110) | - | - | holds |
| G5 six secondary columns | dZ <= 0.10, dM <= 0.30 | dZ <= 0.0163, dM <= 0.0462 | | | holds |

Same subset against all-M private rows: largest bucket 0.0504 / 0.0455 / 0.0486. Extra parity checks that the declared gate does not contain (`v_joint.py`, label-free): joint zero pattern of `it_hist_all_luo` x `seq_cv_pop` by age bucket, row-weighted total-variation distance to private rows 0.2145 at lag 0 and 0.0190 on the private-like view; median / q90 of the non-zero history count of listings older than 76 d: 11 / 86 against 13 / 101 on private rows.

Read (`v_read.py`, run once; own NDCG, own per-feed z and pooling; the same code gives the lead's stored statistic -0.00063 ± 0.00022 on 30,000 feeds):

| quantity, subset, 13,424 feeds | value |
|---|---|
| S_lag (private-like inputs, CPU fp32) | **+0.00024 ± 0.00031** |
| S_0 (lag-0 inputs, CPU fp32) | **-0.00050 ± 0.00032** |
| D = S_lag - S_0 | **+0.00074 ± 0.00040** (1.85 SE; per-feed corr of S_lag and S_0 0.219) |
| S_0 from the stored bf16 files, same feeds | -0.00066 ± 0.00032 |
| rule | S_lag >= -0.0002: true. D >= +1 SE(D): true. S_lag <= -0.0005: false. |
| **branch** | **"bias explanation holds"** |

Robustness of the branch [M]: stored-file S_0 plus D gives +0.00008 (still >= -0.0002). Bootstrap over subset feeds, 2,000 draws: P(S_lag >= -0.0002) = 0.92; P(S_lag <= -0.0005) = 0.007; P(D > 0) = 0.97.

### 1.2 Claims of the study against my numbers

| claim | study's number | my number | agree |
|---|---|---|---|
| test lags, public | 8,562 feeds; 6 d: 6,050; 7 d: 2,512; mean 6.79 d | same | yes |
| test lags, private | 20,033 feeds; 7 d: 4,855; 8 d: 7,578; 9 d: 7,600; mean 8.76 d | same | yes |
| lags imposed on holdout feeds | private-like 7,356 / 11,362 / 11,282; public-like 21,283 / 8,717 | identical arrays (my re-typing of the hash) | yes |
| lag-0 semantics reproduce the stored columns | 7 columns bit-exact, 2 within 5e-7 (4,630,966 rows) | `xp_imp_c50` max diff 2.4e-7 (all rows); `it_hist_*` exact (91,207 + 64,645 sampled rows); `seq_cv_pop` exact, `seq_cv_invd20` / `seq_cv_norm` within 4e-6 (64,645 rows) | yes |
| lagged `xp_imp_c50` (v3), both views | file | my full rebuild equal on 4,630,966 rows | yes |
| lagged `it_hist_*` (v3), both views | files | my rebuild equal on 600 + 400 sampled feeds (exact) | yes |
| lagged `seq_cv_*` (v3), both views | files | my rebuild equal on 400 sampled feeds (`seq_cv_pop` exact, others within 2e-6) | yes |
| gate private: `it_hist_all_luo` dZ lag 0 / lagged / bucket / dM | 0.2474 / 0.0131 / 0.035 / 0.031 | 0.2474 / 0.0131 / 0.0352 / 0.0314 | yes |
| gate private: `seq_cv_pop` | 0.1473 / 0.0090 / 0.030 / 0.025 | 0.1473 / 0.0090 / 0.0304 / 0.0247 | yes |
| gate private: `xp_imp_c50` | 0.2414 / 0.0362 / **0.116 FAIL** / 0.079 | 0.2414 / 0.0362 / **0.1161 FAIL** / 0.0786 (zero share 0.246 against 0.130) | yes |
| gate private: six secondary | dZ <= 0.0144, dM <= 0.045 | dZ <= 0.0144, dM <= 0.0453 | yes |
| gate public: `it_hist_all_luo` / `seq_cv_pop` / `xp_imp_c50` largest bucket | 0.043 / 0.017 / 0.0999 (pass) | 0.0426 / 0.0174 / 0.0999 (pass) | yes |
| gate history v1 (private) | `it_hist` bucket 0.101; `xp_imp` dZ 0.0617, bucket 0.120 | 0.1010; 0.0617, 0.1196 (study's v1 files, my gate code) | yes |
| J(g), g = 0..6 | 0.1452, 0.1294, 0.0941, 0.0693, 0.0989, 0.1771, 0.2659 | identical; argmin g = 3 | yes |
| gate v2 at g = 3, `xp_imp_c50` bucket | 0.193 / 0.156 | 0.1929 / 0.1557 | yes |
| `xp_imp_c50` 10-24 d residual by holdout day | +0.234, +0.115, +0.072, +0.035 | +0.234, +0.115, +0.072, +0.035 | yes |
| subset, private: three primaries dZ / bucket | 0.0156 / 0.050; 0.0145 / 0.045; 0.0225 / 0.049 | 0.0156 / 0.0496; 0.0145 / 0.0447; 0.0225 / 0.0492 | yes |
| feeds with cold share >= 0.45 | 1.7% stored; 57.3% private-like; 50.9% private test | 1.7%; 57.3%; 50.9% | yes |
| control: largest member difference CPU - stored | 0.00018 (30,000 feeds) | 0.00018 | yes |
| control: W8 pool / pop1kd pool, stored and CPU | 0.32456 / 0.32449; 0.31715 / 0.31708 | same | yes |
| control: statistic stored, CPU, difference | -0.00063 ± 0.00022; -0.00047 ± 0.00023; +0.00016 ± 0.00011 | same | yes |
| control: w 0.1 and w 0.3 | -0.00027 / -0.00015; -0.00104 / -0.00093 | same | yes |
| label-free shift, W8 pool | corr 0.9810; top-1 changed 40.5% ± 0.3%; top-10 overlap 0.784 | 0.9810; 40.46% ± 0.28%; 0.7837 | yes |
| label-free shift, pop1kd pool | 0.9942; 18.2% ± 0.2%; 0.897 | 0.9942; 18.20% ± 0.22%; 0.8968 | yes |
| corr(W8 pool, pop1kd pool) | 0.9765 -> 0.9795 (+0.00306 ± 0.00005) | same | yes |
| staged private-like predictions | 12 files, unread | all 12 members re-scored on 400 subset feeds with my nine columns: max diff 1.5e-5, top-1 changed in 0 feeds | yes |
| no labelled read by the study | "read.py read and read_late.py were never run" | no `read_*.txt` log exists; the build, gate, inference and diagnostic scripts open no target file | no evidence against |
| no GPU process, no read-only file changed | as stated | 32 GPU processes, all `train.py` lanes; `lane.sh` 22:14:15Z, `train.py` 20:20:00Z, nothing under `src/`, `feat_new/*.py`, `subs/`, `nn/` after 21:50Z | yes |
| the one self-reported breach | `feat_new/__pycache__/hz_common.cpython-312.pyc`, 12,222 bytes | present, dated 22:22Z, nothing else | yes |

### 1.3 No-decision extras of the 01:08:21Z block [M]

Private-like view (`logs/v_read_priv.txt`):

| quantity | subset, 13,424 feeds | complement, 16,576 feeds (gate fails) | all 30,000 (gate fails) |
|---|---|---|---|
| S_lag, w 0.2 | +0.00024 ± 0.00031 | +0.00017 ± 0.00028 | +0.00020 ± 0.00021 |
| S_0, w 0.2 | -0.00050 ± 0.00032 | -0.00045 ± 0.00032 | -0.00047 ± 0.00023 |
| D, w 0.2 | +0.00074 ± 0.00040 | +0.00062 ± 0.00040 | +0.00067 ± 0.00029 |
| w 0.1: S_lag / S_0 / D | +0.00013 ± 0.00021 / -0.00020 ± 0.00021 / +0.00033 ± 0.00028 | +0.00020 ± 0.00019 / -0.00011 ± 0.00022 / +0.00031 ± 0.00028 | +0.00017 ± 0.00014 / -0.00015 ± 0.00015 / +0.00032 ± 0.00020 |
| w 0.3: S_lag / S_0 / D | -0.00015 ± 0.00038 / -0.00095 ± 0.00041 / +0.00080 ± 0.00047 | +0.00025 ± 0.00036 / -0.00091 ± 0.00039 / +0.00116 ± 0.00049 | +0.00007 ± 0.00026 / -0.00093 ± 0.00028 / +0.00100 ± 0.00034 |
| W8 pool level, lag 0 -> lagged | 0.32334 -> 0.32002: -0.00332 ± 0.00075 | 0.32542 -> 0.32057: -0.00485 ± 0.00086 | 0.32449 -> 0.32032: -0.00417 ± 0.00058 |
| pop1kd pool level, lag 0 -> lagged | 0.31773 -> 0.31724: -0.00049 ± 0.00038 | 0.31656 -> 0.31723: +0.00067 ± 0.00051 | 0.31708 -> 0.31723: +0.00015 ± 0.00033 |
| pop1kd pool - W8 pool, lag 0 / lagged | -0.00561 ± 0.00084 / -0.00278 ± 0.00082 | -0.00886 ± 0.00082 / -0.00334 ± 0.00080 | -0.00741 ± 0.00059 / -0.00309 ± 0.00057 |
| W8 inputs lagged only, change of S | +0.00070 ± 0.00039 | +0.00083 ± 0.00040 | +0.00077 ± 0.00028 |
| `xp_imp_c50` lagged only, change of S | -0.00010 ± 0.00016 | +0.00014 ± 0.00021 | +0.00003 ± 0.00014 |

Members, subset, lagged minus lag 0: the eight W8 members -0.00253 to -0.00412 (SE 0.00071-0.00080 each); the four pop1kd members +0.00001 to -0.00090 (SE 0.00037-0.00040).

By imposed lag, subset: L = 7 (3,271 feeds) S_lag -0.00024 ± 0.00063, D -0.00004 ± 0.00081; L = 8 (5,060) +0.00058 ± 0.00053, D +0.00097 ± 0.00064; L = 9 (5,093) +0.00021 ± 0.00048, D +0.00101 ± 0.00065.

Base pool halves, subset (post-hoc, `v_extra2.py`): kd05w192 x4 as the base pool S_0 -0.00009 ± 0.00032, S_lag +0.00022 ± 0.00033, D +0.00031 ± 0.00040; kd05t25w192 x4 S_0 -0.00050 ± 0.00032, S_lag -0.00002 ± 0.00028, D +0.00047 ± 0.00037; W8 with pop1kd s951-952 only D +0.00072 ± 0.00039, with s953-954 only D +0.00052 ± 0.00040. D is positive in every split and never above 2 SE.

By cold share of the feed (share of rows with lagged `it_hist_all_luo` = 0), subset: [0, 0.35) 1,443 feeds D -0.00040 ± 0.00109; [0.35, 0.45) 4,331 feeds D +0.00100 ± 0.00068; [0.45, 0.55) 4,903 feeds D +0.00067 ± 0.00066; [0.55, 1] 2,747 feeds D +0.00103 ± 0.00093.

Public-like view, subset only (my own CPU scoring of the 12 members on the 13,424 subset feeds, `v_infer_full.py`; imposed lags 6 d on 9,476 and 7 d on 3,948 of them; `logs/v_read_pub_subset_only.txt`). The public-like gate holds on the subset with my code (largest bucket 0.0644 / 0.0334 / 0.0342; dZ <= 0.0237). The complement was NOT scored for this view: I stopped my scoring job at 57 min of extra inference to respect the one-hour cap (6 of 12 members done; the gate fails there anyway: `xp_imp_c50` dZ 0.0528, bucket 0.1552).

| quantity, public-like, subset, 13,424 feeds | value |
|---|---|
| S_lag, w 0.2 | +0.00018 ± 0.00031 |
| D = S_lag - S_0, w 0.2 | +0.00068 ± 0.00039 (1.74 SE) |
| w 0.1: S_lag / D | +0.00019 ± 0.00021 / +0.00039 ± 0.00027 |
| w 0.3: S_lag / D | -0.00033 ± 0.00038 / +0.00062 ± 0.00046 |
| W8 pool level, lag 0 -> lagged | 0.32334 -> 0.32055: -0.00280 ± 0.00074 |
| pop1kd pool level, lag 0 -> lagged | 0.31773 -> 0.31748: -0.00025 ± 0.00036 |
| pop1kd pool - W8 pool, lag 0 / lagged | -0.00561 ± 0.00084 / -0.00307 ± 0.00080 |

My full public-like scoring equals my sample scoring with all nine columns rebuilt by me on the 400 sampled feeds (3 members, max difference 1.5e-5).

Private-like minus public-like on the subset: S +0.00006 ± 0.00018; W8 pool -0.00053 ± 0.00023; pop1kd pool -0.00024 ± 0.00016. By imposed lag, subset, public-like: L = 6 (9,476 feeds) D +0.00104 ± 0.00047; L = 7 (3,948) D -0.00019 ± 0.00070. Taken with the private-like split (L = 7: -0.00004 ± 0.00081; L = 8: +0.00097 ± 0.00064; L = 9: +0.00101 ± 0.00065) there is no visible dose-response in the lag length; the two views differ little because under v3 the lag length only moves listings younger than about 24 d [I].

### 1.4 Post-hoc splits (mine, not pre-declared, no decision) [M]

Where the dense window of a row lies in time under v3 (label-free; `v_extra.py`). Counted dense days = data days before D outside the ±2-day band.

| holdout days | counted (row, dense day) pairs that are AFTER the row | rows with any later day counted | closest counted past day for old listings |
|---|---|---|---|
| Sep 6-8 | 100% (every non-empty window is later days only) | 60-62% | none |
| complement, Sep 6-12 | 81.4% | 59.2% | 3 d before (Sep 9-12) |
| Sep 13-14 | 31.5% | 54.6% | 3 d before |
| Sep 17-19 | 0% | 0% | 3-5 d before |
| subset, Sep 13-19 | 9.9% | 20.9% | 3-5 d before |
| all 30,000 | 42.7% | 41.9% | - |

A private test row has no dense day after it, and its closest dense day is 7-9 d before. Event level, 600 random holdout feeds: 50.0% of the lagged `it_hist_all_luo` count mass comes from dense snapshots dated after the row (`logs/v_ithist.log`).

| group (w 0.2, private-like) | feeds | S_0 | S_lag | D | W8 pool change | pop1kd pool change |
|---|---|---|---|---|---|---|
| Sep 13-14 | 5,145 | -0.00103 ± 0.00057 | +0.00052 ± 0.00053 | +0.00154 ± 0.00072 | -0.00553 ± 0.00138 | -0.00077 ± 0.00067 |
| Sep 17-19 | 8,279 | -0.00017 ± 0.00039 | +0.00007 ± 0.00038 | +0.00024 ± 0.00046 | -0.00195 ± 0.00086 | -0.00031 ± 0.00045 |
| Sep 6-8 | 7,823 | -0.00059 ± 0.00045 | +0.00034 ± 0.00043 | +0.00092 ± 0.00060 | -0.00618 ± 0.00129 | +0.00221 ± 0.00088 |
| Sep 9-12 | 8,753 | -0.00032 ± 0.00046 | +0.00002 ± 0.00036 | +0.00034 ± 0.00054 | -0.00367 ± 0.00114 | -0.00070 ± 0.00057 |

D(Sep 13-14) - D(Sep 17-19) = +0.00130 ± 0.00086 (unpaired; 1.5 SE, not a finding). Stored bf16 statistic by the same groups: Sep 13-14 -0.00105 ± 0.00055; Sep 17-19 -0.00041 ± 0.00039; Sep 6-8 -0.00070 ± 0.00044; Sep 9-12 -0.00051 ± 0.00044.

Is the change specific to the pop-light view? (`v_extra3.py`, post-hoc.) Base pool = one half of W8 (4 members); admixture at w 0.2 of the pop1kd pool (S_pop) or of the other W8 half, which has the same inputs (S_same); mean of the two choices of base half for the last column.

| group | S_pop - S_same at lag 0 (base kd05w192 / base kd05t25w192) | S_pop - S_same lagged | change of S_same under lag | change of (S_pop - S_same) under lag, mean of both bases |
|---|---|---|---|---|
| subset, 13,424 feeds | -0.00058 ± 0.00030 / -0.00068 ± 0.00034 | +0.00017 ± 0.00031 / -0.00010 ± 0.00031 | -0.00043 ± 0.00026 / -0.00011 ± 0.00025 | **+0.00066 ± 0.00031** |
| all 30,000 feeds | -0.00033 ± 0.00021 / -0.00071 ± 0.00023 | +0.00023 ± 0.00021 / -0.00021 ± 0.00021 | -0.00031 ± 0.00018 / -0.00007 ± 0.00018 | **+0.00053 ± 0.00023** |

Reading [I]: a same-view admixture does not gain under the lag, the pop-light one does (2.1 SE on the subset). So D is not the generic effect "any second pool helps a degraded pool". Under the lag the pop-light quartet is worth about as much as four more full-view students at w 0.2 (difference +0.00017 / -0.00010), not more.

## 2. Issues, by severity

**High (interpretation; none overturns the study's verdict).**
1. **The "lagged" view is not a view from the past.** For listings older than 24 d (53% of rows) D = Sep 20: the row keeps every dense day except the five days around it, including the days after it. The study writes this in `DECLARATION.txt` ("old listings are NOT made stale") and names the calibration in its limits, but the report's headline terms ("lagged", "private-like", "test-like inputs") and the gate do not show that staleness for old listings is 3 d instead of 7-9 d and that early holdout days are fed from the future. The gate compares zero share and mean by item age and cannot see direction in time. Labelled trace of the artefact: on Sep 6-8 the pop1kd pool IMPROVES under the "lag" by +0.00221 ± 0.00088 (7,823 feeds), because its stored `xp_imp_c50` is causal (zero on the first days) and the rebuilt one is filled from later days. Consequence: the complement is not a valid emulation for a reason the failing criterion reflects only indirectly; a repair of `xp_imp_c50` alone (the study's option 2) would not make the 30,000-feed read valid.
2. **Direction of the residual bias on the subset is not established.** Counters fed partly from later days (Sep 13-14) favour the W8 pool more than a real test row would, which makes S_lag conservative [I]; but D is largest exactly there, and near zero on the causal days Sep 17-19, where S_0 itself is near zero. Both facts fit the bias explanation in the form "the in-period penalty comes from counters that see nearby and later days". They also mean the decision rests mostly on 5,145 feeds.
3. **The gate was rebuilt twice after its result was seen and one parameter was fitted on the gate's own criterion** (g = 3 minimises the summed dZ of the decision view). Parity of the eight W8 inputs (dZ 0.013) is therefore partly by construction and is not an independent check. Disclosed by the study. Under a strict reading the gate result is v1: private FAIL on three counts. Same outcome: no read.

**Medium.**
4. The public-like PASS on 30,000 feeds is not a pass in substance: G2 holds by 0.0001 (0.0999), and the 7-10 d bucket of `xp_imp_c50` has a gap of 0.2021 that G2 skips only because the bucket holds 4.8% of rows, under the 5% floor. On the complement the public-like view fails G1 and G2 (0.0528 / 0.1552). The study says it would not lean on that pass; I agree and would report it as "not passed".
5. **The gate covers only the nine rebuilt columns.** The other 26 inputs were declared recency-free from a code read. On data (`v_other.py`, label-free): on the subset their largest standardised mean gap to private rows is 0.171 SD (`ctx_nd_knn_u`), the rest <= 0.16; on the complement the pre-trained-encoder inputs (`pre2_*`) are 100% missing on Sep 6-8 (7,823 feeds, `has:pre2_fe` = 0), a format no test row has (known to the lead as HOLD-CLEAN step 2). One more reason why only the subset is test-like.
6. The fixed rule has little resolution at this sample size: SE of S_lag on the subset is 0.00031, the two thresholds are 0.0003 apart. With D >= +1 SE(D) added, the rule returns "holds" in 7.6% / 6.3% / 4.0% of cases when the lag changes nothing and the true S is -0.00050 / -0.00055 / -0.00066 [I: bivariate normal with the measured SEs and the measured per-feed correlation 0.219, 2,000,000 draws]. For the observed data the bootstrap gives P(S_lag >= -0.0002) = 0.92 and P(D > 0) = 0.97: the result is not at the edge of the branch.
7. Rule breach, self-reported and confirmed: `feat_new/__pycache__/hz_common.cpython-312.pyc` created at 22:22Z. No executed file changed. Left in place.
8. Timeline of the declarations: the file's modification time equals the stamp of its last block (22:52:46Z); `build_xp3.py` carries the same second and the v3 column was written 10 s later. The stamps of the first two blocks precede their builds by 3-6 min according to the script and column times; they cannot be checked beyond that. `gate.py` was edited at 22:55:46Z, after the final gate (22:53:02Z), and re-run (`gate_v3check.txt`); my own gate code gives the same numbers, so the verdict does not depend on that edit.

**Low.**
9. Sparse-day rule of `seq_cv_*`: the declaration says "not a counted dense snapshot", the code uses "a day outside the counted dense window". The two readings differ in `seq_cv_pop` on 1.2% of rows (private-like) and 0.6% (public-like) of my sample. Immaterial.
10. Short jobs next to the 12-thread inference and the staging of predictions after the FAIL: outside the letter of the rules, harmless, and the staging made the subset read possible.
11. SEs in the study's scripts are paired per feed (standard deviation of per-feed differences over sqrt(n)): correct.
12. Leak audit: no target file is opened by `build_lag*.py`, `build_xp3.py`, `gate*.py`, `infer.py`, `diag.py`; test rows enter through input columns only; later snapshots of the same non-zero user are removed in `it_hist_*` and `xp_imp_c50`, same-user sequences are skipped in `seq_cv_*` (my rebuild with those rules reproduces the files). User 0 (5.1% of holdout feeds) cannot be separated, as in the stored columns. No leak found.

## 3. What the lead should conclude

- The study's verdict stands: no read on the 30,000 feeds, control valid, numbers correct.
- The subset read returns "bias explanation holds". Read it as: under test-like coverage of the popularity inputs the pop-light admixture stops costing the W8 pool anything (S_lag +0.00024 ± 0.00031 against S_0 -0.00050 ± 0.00032), because the W8 pool loses -0.00332 ± 0.00075 and the pop1kd pool -0.00049 ± 0.00038. It is not evidence that the admixture gains out of period: S_lag at w 0.1 / 0.2 / 0.3 is +0.00013 / +0.00024 / -0.00015, all within 1 SE of zero.
- The lagged-test condition of amendment (9) is met in the form the 01:08:21Z block gives it: W8P20 is eligible as pick 2, and as pick 1 only if the public read gives >= W8 + 0.0008 (A'). The lagged test alone moves no pick.
- The effect is specific to the pop-light view (against a same-view admixture: +0.00066 ± 0.00031 on the subset under the lag), but under the lag the pop1kd quartet is worth about what four more full-view students are worth at w 0.2, not more [M, post-hoc]. The lagged holdout gives no reason to expect the public read of W8P20 clearly above W8 (public-like subset S_lag +0.00018 ± 0.00031, train-only level, feeds with M >= 50 only) [I].
- If a firmer reading is wanted: only holdout days Sep 17-19 give a view from the past, with old listings stale by 3-5 d instead of 7-9 d (8,279 feeds; SE of D 0.00046). A causal view with a 7-9 d lag and test-like depth cannot be built from this data, because the dense data start on Sep 6 (the known T4 depth collapse) [I].
- The change of the statistic comes from the eight W8 inputs (+0.00070 ± 0.00039 on the subset, +0.00077 ± 0.00028 on all feeds), not from `xp_imp_c50` (-0.00010 ± 0.00016; +0.00003 ± 0.00014). The criterion that failed on 30,000 feeds concerns a column that does not move the statistic. The reason to distrust the 30,000-feed number is issue 1 (and issue 5 for Sep 6-8), not that criterion.

## 4. Method

- **Holdout identity** re-derived from `nn_x/w5/train.py` (p_random > 0.9, M >= 50, rng 12345): equal to `holdout_meta.npy`, to the study's mini split and to the train targets of those rows.
- **NDCG / z / pools**: my own functions in `v_common.py` (per-feed full-list NDCG, binary gain, negatives first among ties; z = (x - feed mean) / (feed sd + 1e-9); pool = member mean of z). Checked against a plain loop on 1,500 feeds and against `src/metrics.py` on 30,000 feeds (max difference 0), and by reproducing -0.00063 ± 0.00022 from the stored files.
- **`xp_imp_c50`**: rebuilt for all 4,630,966 rows and both views from the text of the declaration (one pass over the train candidate item ids for per-day impressions; polars join for the own and later rows of the same user).
- **`it_hist_*`**: event-level join of every history event of the sampled items in train, val and test with the sampled rows; stored rule and v3 rule; 600 random holdout feeds and 400 random subset feeds.
- **`seq_cv_*`**: my own numpy implementation written from `feats_v2/seq_/feat_covis.py` and the membership text of the declaration (no numba, no code of the study), validated on the stored lag-0 columns first, then compared with the study's v3 files on the 400 subset feeds. My first run missed the stored rule "empty history -> pop = 0" (my error, 2.6% of rows; `logs/v_sample_run1_popbug.log`); after the fix `seq_cv_pop` is exact.
- **Scoring**: my own sample split cut from `data/flat`, the lead's `SplitData.gather / batches / scatter` and `Ranker` (imported, bytecode writing disabled), fp32 on CPU, domain mix [1 - p_random, p_random, 0]. All 12 members on the 400 subset feeds with my nine columns against the staged files; 2 members on 1,000 random holdout feeds in the first run.
- **Subset gate**: `v_subgate.py`, criteria G1-G5 as written in the study's declaration, subset rows as the holdout population, private test rows of feeds with M >= 50 as the reference.
- **Read**: `v_read.py`. Lagged predictions = the study's staged files; lag-0 predictions = the study's CPU replay (both confirmed as above). One run; the extras were printed by the same run. `v_extra.py` (day splits, window direction) was written and run after the decision read was on disk.
- **Public-like view**: the subset scored by me (`v_infer_full.py`), 12 members, `it_hist_*` / `seq_cv_*` from the study's `lag_pub_v3_*` files (confirmed on the samples), `xp_imp_c50` from my rebuild.

## 5. Limits and what I did NOT check

- The full 30,000-feed `it_hist_*` and `seq_cv_*` arrays were confirmed on samples (600 random holdout feeds / 91,207 rows plus 400 subset feeds / 64,645 rows for `it_hist_*`; the 400 subset feeds for `seq_cv_*`), not rebuilt in full. The staged predictions were confirmed on 400 subset feeds for all 12 members, not re-scored in full.
- The v1 and v2 column files were not rebuilt; I only re-applied the gate to them.
- The public-like view on the complement and on all 30,000 feeds: not scored (one-hour cap).
- One draw of the lag hash. I tested the `xp_imp_c50` gate under five other draws (largest bucket 0.1160-0.1163 private, 0.0998-0.0999 public); I did not re-score under another draw, so the draw-to-draw spread of S_lag is unknown.
- Accuracy of the item-age estimate from f4 on holdout against test rows: not tested.
- No GPU bf16 replay of lagged inputs. The CPU level of S is +0.00016 ± 0.00011 above the bf16 level at lag 0; D is CPU against CPU.
- Not covered: `@last` checkpoints, fine-tuned members, the 16-member pool, the wide pop1 students (their lag-0 statistic is -0.00050 ± 0.00022 against W8 by the lead's 01:06Z block), hard-label pop1 members.
- I cannot prove that nobody looked at a lagged NDCG before the 01:08:21Z declaration; I found no trace of one.
- The 4-8% false-pass figure comes from a bivariate normal with the measured SEs, not from a resampling of feeds.

## 6. Commands to reproduce

```
cd <repo>/nn_x/w12/next/poplag_verify
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
PY=<repo>/.venv/bin/python
$PY v_control.py                 # control with own NDCG / pools              -> logs/v_control.log
$PY v_gate.py                    # lags, full xp_imp_c50 rebuild, gate v3      -> logs/v_gate.log, cache/my_{priv,pub}_v3_xp_imp_c50.npy
$PY v_ithist.py 600              # it_hist rebuild on 600 feeds, window origin -> logs/v_ithist.log
PV_THREADS=4 $PY v_sample.py 400 # nine columns on 400 subset feeds + scoring  -> logs/v_sample.log, cache/sample_*
PV_THREADS=4 $PY v_sample12.py   # all 12 members on that sample               -> logs/v_sample12.txt
PV_THREADS=4 $PY v_subgate.py    # G1-G5 on subset / complement / all          -> logs/v_subgate.log
PV_THREADS=4 $PY v_read.py       # THE SUBSET READ + extras (private-like)     -> logs/v_read_priv.txt
PV_THREADS=4 $PY v_extra.py      # post-hoc day splits, window direction       -> logs/v_extra.txt
PV_THREADS=4 $PY v_claims.py     # gate history, cold share, prediction shift  -> logs/v_claims.txt
PV_THREADS=4 $PY v_joint.py      # joint parity, count magnitudes              -> logs/v_joint.txt
PV_THREADS=4 $PY v_other.py      # the 26 inputs left as stored                -> logs/v_other.txt
PV_THREADS=4 $PY v_extra2.py     # post-hoc: base-pool halves                  -> logs/v_extra2.txt
PV_THREADS=4 $PY v_extra3.py     # post-hoc: pop-light vs same-view admixture  -> logs/v_extra3.txt
$PY v_infer_full.py pub_v3 late  # public-like scoring of the subset, 12 members (30 min) -> cache/pred_pub_v3_late_*.npy
PV_THREADS=4 $PY v_read.py pub   # public-like extras, subset                  -> logs/v_read_pub_subset_only.txt
# not finished: $PY v_infer_full.py pub_v3 early (complement; about 45 min; then v_read.py pub prints complement and all -> logs/v_read_pub.txt)
```

## 7. Files created

All under `nn_x/w12/next/poplag_verify/` unless stated.
- `PROGRESS.md`; scripts `v_common.py`, `v_control.py`, `v_gate.py`, `v_ithist.py`, `v_infer.py`, `v_sample.py`, `v_sample12.py`, `v_subgate.py`, `v_read.py`, `v_extra.py`, `v_claims.py`, `v_joint.py`, `v_other.py`, `v_extra2.py`, `v_extra3.py`, `v_infer_full.py`, `run_pub.sh`.
- Logs: `logs/v_control.log`, `logs/v_gate.log`, `logs/v_ithist.log`, `logs/v_infer.log`, `logs/v_sample.log`, `logs/v_sample_run1_popbug.log`, `logs/v_sample12.txt`, `logs/v_subgate.log`, `logs/v_read_priv.txt`, `logs/v_extra.txt`, `logs/v_claims.txt`, `logs/v_joint.txt`, `logs/v_other.txt`, `logs/v_extra2.txt`, `logs/v_extra3.txt`, `logs/v_read_pub_subset_only.txt`, `logs/v_infer_full_pub_{late,early}.log`, `logs/run_pub.log`.
- Cache: `cache/my_{priv,pub}_v3_xp_imp_c50.npy` (full rebuild), `cache/sample_*` (sample columns, sample predictions, sample index), `cache/pred_pub_v3_late_<member>.npy` (12 members), `cache/pred_pub_v3_early_<member>.npy` (6 of 12 members, unused), `cache/v_read_pub_subset_only.json`, `cache/v_gate.json`, `cache/v_subgate.json`, `cache/v_sample.json`, `cache/v_read_priv.json`.
- `reports/NEXT_1115/poplag_verify.md` (this file).

## 8. Compliance

- CPU only, `CUDA_VISIBLE_DEVICES=""` in every command, at most 12 threads per process. One heavy process at a time (the public-like scoring, PIDs recorded in `PROGRESS.md`). While it ran I started short helper jobs capped at 4 threads (`PV_THREADS=4`; 15-75 s each): the same judgment call the study flagged for itself.
- Processes stopped: only my own two, by exact PID after reading `/proc/<pid>/cmdline` (wrapper 3909581 `bash ./run_pub.sh`, then 3955377 `python v_infer_full.py pub_v3 early`), at 02:10Z, to keep the extra inference near the one-hour cap. Nothing of mine is left running. Nothing written outside `nn_x/w12/next/poplag_verify/` and this report. The lead's modules were imported with bytecode writing disabled. No install, no network, no submission file.
- Labels: the holdout labels were opened by `v_control.py` (lag 0 only), `v_read.py`, `v_extra.py`, `v_extra2.py` and `v_extra3.py`. No val or test label was opened. Test rows were used through input columns only.
