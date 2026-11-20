# u20wit: U20 read-out by user fold + a label-clean transductive witness

Written 2026-11-15 03:50 UTC by the u20wit agent (CPU only; resumed run 01:08-03:50 UTC). All six runs `w5_e10_u20{a,b,c}_s95{1,2}` are finished (last json 03:11:14 UTC).
± = 1 SE paired per feed; n = feeds. [M] measured by me, [I] inferred. "Member delta" = per-feed mean over the seeds of NDCG(B, seed) - NDCG(A, seed); "2-pool" = NDCG of the mean per-feed rank-z of the two seeds.

## Verdict

1. **U20 by the fixed rule on the CROSS-USER columns (seeds 951, 952): 2 MORE SEEDS** (neither GO nor KILL) [M]. B - A holdout (n 24,026) = +0.00061 ± 0.00038 selected, +0.00040 ± 0.00038 @last (bars: GO >= +0.0008, KILL < +0.0004; @last is +0.0004026, i.e. on the KILL line); B - A val (n 13,571) = +0.00099 ± 0.00064; C - A holdout = +0.00143 ± 0.00040; ratio 0.43 selected / 0.30 @last (bar 0.5).
2. At 2-pool level the holdout shows no gain for B: 2-pool B - A = -0.00007 ± 0.00044 (C - A 2-pool +0.00073 ± 0.00045); the member delta per seed is +0.00001 / +0.00121. On val the 2-pool reads +0.00120 ± 0.00074 [M].
3. **Transductive witness** (exploration-like train feeds of fold-0 users: unlabelled inputs of B, never seen by A; CPU vs CPU, selected): B - A = **+0.00060 ± 0.00082 on short feeds** (M < 50, n 11,022; +0.00070 ± 0.00084 with the val / test domain token) and **+0.00095 ± 0.00037 on long feeds** (M >= 50, n 26,883); 2-pool -0.00038 ± 0.00097 / +0.00026 ± 0.00042; @last +0.00037 ± 0.00083 / +0.00086 ± 0.00037 [M].
4. Whether a feed's own inputs add anything beyond the general effect of the unlabelled fold is not established: long witness minus cross-user holdout (CPU, n 8,000) = +0.00013 ± 0.00076, short minus the same = -0.00022 ± 0.00106; long witness minus the same users' unseen feeds (H0, n 5,974) = +0.00182 ± 0.00084 (@last +0.00122 ± 0.00084). B matches the teacher by the same extra amount on seen and unseen long feeds (within-feed corr +0.0064 vs +0.0065) [M]. In-sample bound C - A on the same feeds: +0.0229 ± 0.0009 short / +0.0178 ± 0.0004 long (bound only, not evidence).
5. The witness and the CPU path were validated on known answers before any U20 number was read: CPU vs teacher_xf0 within-feed corr 0.99994; base - xf0 = +0.0237 / +0.0171 on the fold-0 witness against +0.0025 / +0.0012 on fold-1 controls; positives per feed within 0.04 of val M < 50 in every size bin [M].
6. Next step by the rule: seeds 953, 954 of u20a and u20b (u20c too if the ratio is to be read). Write the 4-seed rule before they finish: at a true +0.0006 a 4-seed read stays between the bars with P about 0.66 [M, A/A null of same-recipe students]. Read-out command: `SEEDS=951,952,953,954 bash nn_x/w12/next/u20wit/run_seeds.sh`.

## 1. Measurements

### 1.1 Declared read-out (stored GPU train-only predictions, GPU files vs GPU files; `out/u20_read.txt`)

Rule (20:21Z block + 21:26Z addendum, thresholds unchanged): GO iff B - A >= +0.0008 holdout AND >= 0 val AND (B - A)/(C - A) >= 0.5; KILL iff B - A < +0.0004 holdout; otherwise 2 more seeds; cross-user columns only.

| column | n | checkpoint | B - A member (per seed 951, 952) | B - A 2-pool | C - A member (per seed) | C - A 2-pool | ratio, member (bootstrap 68%) |
|---|---|---|---|---|---|---|---|
| **holdout, cross-user** | 24,026 | selected | **+0.00061 ± 0.00038** (+0.00001, +0.00121) | -0.00007 ± 0.00044 | +0.00143 ± 0.00040 (+0.00105, +0.00180) | +0.00073 ± 0.00045 | 0.43 (0.19 .. 0.66) |
| | | @last | **+0.00040 ± 0.00038** (-0.00040, +0.00121) | -0.00036 ± 0.00044 | +0.00133 ± 0.00040 (+0.00086, +0.00180) | +0.00041 ± 0.00045 | 0.30 (0.00 .. 0.54) |
| **val, cross-user** | 13,571 | selected | **+0.00099 ± 0.00064** (+0.00181, +0.00017) | +0.00120 ± 0.00074 | +0.00096 ± 0.00066 (+0.00138, +0.00054) | +0.00092 ± 0.00078 | 1.03 (0.39 .. 1.76) |
| | | @last | +0.00085 ± 0.00064 (+0.00152, +0.00017) | +0.00087 ± 0.00074 | +0.00094 ± 0.00066 (+0.00133, +0.00054) | +0.00093 ± 0.00079 | 0.90 (0.21 .. 1.65) |
| holdout, fold-0 users | 5,974 | selected | -0.00111 ± 0.00075 (-0.00083, -0.00140) | -0.00057 ± 0.00083 | +0.00055 ± 0.00083 (+0.00049, +0.00062) | +0.00033 ± 0.00092 | not usable (C - A within 1 SE of 0) |
| | | @last | -0.00054 ± 0.00075 (+0.00032, -0.00140) | -0.00106 ± 0.00083 | +0.00106 ± 0.00082 (+0.00150, +0.00062) | +0.00023 ± 0.00092 | not usable |
| val, fold-0 users | 852 | selected | +0.00483 ± 0.00237 (+0.00756, +0.00210) | +0.00763 ± 0.00265 | +0.00200 ± 0.00219 (+0.00180, +0.00220) | +0.00200 ± 0.00266 | not usable |
| | | @last | +0.00546 ± 0.00235 (+0.00882, +0.00210) | +0.00717 ± 0.00261 | +0.00244 ± 0.00224 (+0.00269, +0.00220) | +0.00325 ± 0.00267 | not usable |
| holdout, all | 30,000 | selected | +0.00027 ± 0.00034 (-0.00015, +0.00069) | -0.00017 ± 0.00039 | +0.00125 ± 0.00036 (+0.00094, +0.00157) | +0.00065 ± 0.00040 | 0.21 (-0.07 .. 0.44) |
| | | @last | +0.00021 ± 0.00034 (-0.00026, +0.00069) | -0.00050 ± 0.00039 | +0.00128 ± 0.00036 (+0.00098, +0.00157) | +0.00038 ± 0.00041 | 0.17 (-0.14 .. 0.38) |
| val, all | 14,423 | selected | +0.00122 ± 0.00062 (+0.00215, +0.00029) | +0.00158 ± 0.00071 | +0.00102 ± 0.00064 (+0.00140, +0.00064) | +0.00098 ± 0.00075 | 1.19 (0.57 .. 2.10) |
| | | @last | +0.00112 ± 0.00062 (+0.00195, +0.00029) | +0.00125 ± 0.00072 | +0.00103 ± 0.00064 (+0.00141, +0.00064) | +0.00106 ± 0.00076 | 1.09 (0.50 .. 1.79) |
| val, cross-user, M < 50 | 5,541 | selected | +0.00022 ± 0.00122 (+0.00141, -0.00097) | +0.00101 ± 0.00144 | +0.00103 ± 0.00124 | +0.00046 ± 0.00146 | - |
| val, cross-user, M >= 50 | 8,030 | selected | +0.00152 ± 0.00067 (+0.00208, +0.00096) | +0.00134 ± 0.00076 | +0.00091 ± 0.00072 | +0.00123 ± 0.00086 | - |

Rule applied [M]: selected: +0.00061 is < +0.0008 (no GO; the ratio 0.43 < 0.5 would block GO as well) and >= +0.0004 (no KILL) -> 2 MORE SEEDS. @last: +0.0004026 >= +0.0004 (no KILL, by 0.0000026) and < +0.0008 -> 2 MORE SEEDS. Both checkpoint choices give the same verdict; the 20:21Z text names both and does not say which one decides.

Levels, selected checkpoints, per seed (951, 952) and 2-pool [M]:

| column | n | u20a | u20b | u20c |
|---|---|---|---|---|
| val, fold-0 users | 852 | 0.38301, 0.38861; pool 0.38736 | 0.39057, 0.39071; pool 0.39499 | 0.38481, 0.39081; pool 0.38935 |
| val, cross-user | 13,571 | 0.39702, 0.39768; pool 0.39863 | 0.39882, 0.39785; pool 0.39984 | 0.39840, 0.39822; pool 0.39955 |
| val, all | 14,423 | 0.39619, 0.39714; pool 0.39797 | 0.39833, 0.39743; pool 0.39955 | 0.39759, 0.39778; pool 0.39895 |
| holdout, fold-0 users | 5,974 | 0.31948, 0.31930; pool 0.31961 | 0.31866, 0.31790; pool 0.31904 | 0.31997, 0.31993; pool 0.31994 |
| holdout, cross-user | 24,026 | 0.32179, 0.32169; pool 0.32333 | 0.32181, 0.32290; pool 0.32326 | 0.32284, 0.32350; pool 0.32406 |
| holdout, all | 30,000 | 0.32133, 0.32122; pool 0.32259 | 0.32118, 0.32191; pool 0.32242 | 0.32227, 0.32279; pool 0.32324 |

Sensitivity of the decision column to its definition (`out/u20_read_sens.txt`, selected / @last) [M]: holdout cross-user without the user-0 bucket (n 22,796) B - A +0.00074 ± 0.00039 / +0.00054 ± 0.00039; by user fold 1..4 (n 6,003 / 6,087 / 5,931 / 6,005, selected) +0.00102 ± 0.00077, +0.00024 ± 0.00075, +0.00013 ± 0.00079, +0.00106 ± 0.00077; val with the strict hash split (all users hashing to fold 0 removed, n 11,727) +0.00106 ± 0.00068 / +0.00087 ± 0.00068; val unseen users only (n 10,037) +0.00106 ± 0.00074 / +0.00080 ± 0.00074. No definition moves the verdict to GO or KILL.

How to weigh it (A/A null of the same statistic from the 8 finished same-recipe students, `out/aa_null_u20_k.txt`) [M]: with 2 seeds per arm the pooled member delta on the cross-user holdout has sd 0.00029 (kd05w192) / 0.00025 (kd05t25w192) under "no effect"; a null draw reaches +0.00061 in 0.7% / 0.0% of 420 draws. The member gain of B over A on the holdout is therefore unlikely to be seed noise, but A has 78,710 steps and B, C 98,350, and the 2-pool does not show it.

### 1.2 Transductive witness (CPU fp32 replays only; `out/u20_wit_read_tr.txt`, `out/xh_read.txt`)

Feeds: train feeds of fold-0 users, XPOOL v2 posterior p_random > 0.9, outside the 30K holdout. In u20a's training: 0 of them. In u20b's training: all, as unlabelled inputs (teacher-only target, K9 aux weight 0), none in its cooldown pool. In u20c's training: all, labelled (C - A is in-sample for C). C levels (member / 2-pool): W0S 0.51768 / 0.51966, W0L 0.33824 / 0.33918, H0 0.31979 / 0.32001.

| feeds | n | A member / 2-pool | B member / 2-pool | **B - A member** (per seed) | B - A 2-pool | B - A member @last | C - A member, IN-SAMPLE bound | B - A as share of the bound |
|---|---|---|---|---|---|---|---|---|
| **W0S short (M < 50)** | 11,022 | 0.49482 / 0.49650 | 0.49542 / 0.49612 | **+0.00060 ± 0.00082** (+0.00126, -0.00006) | -0.00038 ± 0.00097 | +0.00037 ± 0.00083 | +0.02286 ± 0.00089 (2-pool +0.02316 ± 0.00107) | 2.6% |
| M 5-9 | 1,266 | 0.69954 / 0.70198 | 0.69615 / 0.69802 | -0.00339 ± 0.00264 | -0.00396 ± 0.00325 | -0.00237 ± 0.00265 | +0.02345 ± 0.00273 | |
| M 10-19 | 1,872 | 0.56212 / 0.56400 | 0.56082 / 0.55971 | -0.00130 ± 0.00217 | -0.00429 ± 0.00258 | -0.00045 ± 0.00216 | +0.02355 ± 0.00239 | |
| M 20-34 | 3,930 | 0.47476 / 0.47511 | 0.47775 / 0.47894 | +0.00298 ± 0.00141 | +0.00383 ± 0.00162 | +0.00216 ± 0.00143 | +0.02478 ± 0.00150 | |
| M 35-49 | 3,954 | 0.41735 / 0.42002 | 0.41775 / 0.41845 | +0.00041 ± 0.00121 | -0.00157 ± 0.00144 | -0.00012 ± 0.00124 | +0.02044 ± 0.00135 | |
| short, re-weighted to the val M < 50 size mix | eff. 9,366 | | | +0.00011 ± 0.00094 | | +0.00000 ± 0.00095 | | |
| short, scored with the val / test domain token ('rl'; `out/rl_short_read.txt`) | 11,022 | 0.49431 / - | 0.49501 / - | +0.00070 ± 0.00084 (+0.00142, -0.00002) | +0.00002 ± 0.00098 | not scored | not scored | |
| **W0L long (M >= 50)** | 26,883 | 0.32044 / 0.32173 | 0.32139 / 0.32199 | **+0.00095 ± 0.00037** (+0.00131, +0.00059) | +0.00026 ± 0.00042 | +0.00086 ± 0.00037 | +0.01780 ± 0.00042 (2-pool +0.01746 ± 0.00048) | 5.4% |
| long, in C's cooldown pool (old p > 0.5) | 17,462 | 0.32004 / 0.32173 | 0.32103 / 0.32180 | +0.00099 ± 0.00046 | +0.00007 ± 0.00053 | +0.00094 ± 0.00046 | +0.01931 ± 0.00053 | |
| long, not in C's pool | 9,421 | 0.32118 / 0.32172 | 0.32206 / 0.32233 | +0.00088 ± 0.00061 | +0.00061 ± 0.00069 | +0.00073 ± 0.00060 | +0.01498 ± 0.00070 | |
| long, M 50-99 | 9,143 | 0.36302 / 0.36440 | 0.36452 / 0.36532 | +0.00149 ± 0.00072 | +0.00092 ± 0.00085 | +0.00132 ± 0.00072 | +0.02120 ± 0.00081 | |
| long, M 100-300 | 17,740 | 0.29849 / 0.29973 | 0.29917 / 0.29966 | +0.00068 ± 0.00041 | -0.00008 ± 0.00047 | +0.00063 ± 0.00041 | +0.01604 ± 0.00048 | |
| W0S + W0L | 37,905 | 0.37115 / 0.37255 | 0.37200 / 0.37262 | +0.00085 ± 0.00035 (+0.00130, +0.00040) | +0.00008 ± 0.00041 | +0.00072 ± 0.00035 | +0.01927 ± 0.00039 | |
| H0 = fold-0 holdout (inputs seen by nobody; B saw the user's other feeds unlabelled) | 5,974 | 0.31941 / 0.31986 | 0.31855 / 0.31886 | -0.00086 ± 0.00076 (-0.00059, -0.00114) | -0.00100 ± 0.00085 | -0.00036 ± 0.00076 | +0.00038 ± 0.00082 (out-of-sample) | |
| XH = cross-user holdout, random 8,000 of 24,026 (feeds in nobody's training; their users are labelled for A and B alike) | 8,000 | 0.32440 (members 0.32391, 0.32488) | 0.32522 (members 0.32452, 0.32591) | +0.00082 ± 0.00067 (+0.00061, +0.00103); same 8,000 feeds from the GPU files +0.00075 ± 0.00066 | +0.00018 ± 0.00075 | not scored | +0.00100 ± 0.00069 (+0.00142, +0.00058); 2-pool -0.00002 ± 0.00077 (out-of-sample) | |

Baselines for "what did the feed's own inputs add" (independent feed sets, SE = root sum of squares) [M]:

| contrast (B - A, member, selected, CPU vs CPU) | value |
|---|---|
| W0L minus XH (cross-user holdout, n 8,000) | +0.00013 ± 0.00076 |
| W0S minus XH | -0.00022 ± 0.00106 |
| W0L minus H0 (same users, n 5,974) | +0.00182 ± 0.00084 (@last +0.00122 ± 0.00084) |
| W0L with old p_random > 0.9 (the holdout's own selection, n 9,392) minus H0 | +0.00149 ± 0.00100 (@last +0.00086 ± 0.00100) |
| H0 minus XH (same-user channel without the feed) | -0.00168 ± 0.00101 |
| W0L with old p_random > 0.9 minus XH | -0.00020 ± 0.00093 |
| W0S + W0L minus XH | +0.00003 ± 0.00075 |
| CPU minus GPU-file value of the XH delta itself (same 8,000 feeds) | +0.00007 ± 0.00019 |
| C - A: H0 minus XH (both out-of-sample for C) | -0.00062 ± 0.00107 |
| C - A: W0L minus XH, W0S minus XH (in-sample margin of C net of its data slope; bound only) | +0.01680 ± 0.00081, +0.02186 ± 0.00112 |

Mechanism, label-free [M]: mean within-feed correlation to teacher_xf0, B minus A (selected): W0S +0.00769 ± 0.00014, W0L +0.00638 ± 0.00004, H0 +0.00650 ± 0.00009 (levels A 0.9721 / 0.9746 / 0.9744, B 0.9798 / 0.9810 / 0.9809, C 0.9750 / 0.9772 / 0.9775). B is more teacher-like everywhere, and on long feeds by the same amount whether it trained on the feed's inputs (W0L) or not (H0). The teacher itself (CPU mean of its 3 members, which never saw fold 0) reads 0.49511 / 0.32082 / 0.31856 on W0S / W0L / H0, against 0.49482 / 0.32044 / 0.31941 for one A member and 0.49650 / 0.32173 / 0.31986 for the A 2-pool: the teacher of the unlabelled fold is no better than the student it teaches.

Anchor of the CPU path on the U20 checkpoints (H0, CPU minus stored GPU file, per member) [M]: u20a_s951 -0.00022 ± 0.00023, u20a_s952 +0.00026 ± 0.00022, u20b_s951 +0.00002 ± 0.00020, u20b_s952 +0.00053 ± 0.00021, u20c_s951 -0.00019 ± 0.00022, u20c_s952 -0.00013 ± 0.00019. On the 8,000 XH feeds the six members read CPU minus GPU file -0.00006 .. -0.00017 (each within 1 SE). On H0 the B - A member delta is -0.00086 on the CPU and -0.00111 from the GPU files, on XH +0.00082 and +0.00075: the CPU / GPU difference of a delta is 0.0001-0.0003, which is why only CPU numbers are compared with CPU numbers above.

### 1.3 Validation on known answers (run before any U20 read; `out/wit_validate.txt`, re-run byte-identical in the resumed run) [M]

| check | result |
|---|---|
| caches of the earlier (cut) run | feed lists reproduce exactly; 500 random feeds: 0 mismatching raw arrays against the source memmaps; 500 / 300 feeds re-scored: max logit difference 7.6e-06 / 1.5e-05 against the cached scores (`out/recheck_cache.log`) |
| (a) teacher file | `teacher_xf0.npy` = SUM of 3 members (`teacher_xf0.n` = 3; `dump_xf0.log`: w5_e10_xf0_s951, s952, s953), so `--kd_T 3` gives the mean logit |
| (a) CPU mean of the 3 members vs teacher / 3, within-feed corr | W0S 0.999938, W0L 0.999942, H0 0.999946, W1S 0.999936, W1L 0.999943 (medians 0.99997; 1st percentile 0.99943-0.99966). 140 of 61,107 feeds are below 0.999 (min 0.983; RMS logit difference 0.05 against a within-feed spread of 0.9); a single member reads 0.978-0.981; a deliberately misaligned copy reads -0.0007 |
| (a) NDCG of my 3-mean vs the teacher file, paired | W0S +0.00023 ± 0.00024, W0L -0.00004 ± 0.00009, H0 -0.00013 ± 0.00025, W1S -0.00050 ± 0.00023, W1L -0.00000 ± 0.00021 |
| (a) CPU vs stored GPU holdout file on H0 (n 5,974), 9 d128 members and 2 d192 students | paired difference -0.00016 .. +0.00034, each within 1.5 SE; within-feed corr 0.99992-0.99995 |
| (b) base - xf0, member, 3 seeds (base saw fold 0 with labels, xf0 did not) | fold-0 witness: W0S +0.02369 ± 0.00090, W0L +0.01707 ± 0.00042. Fold-1 controls (in-sample for both): W1S +0.00253 ± 0.00086, W1L +0.00120 ± 0.00087. H0 (out-of-sample for both): +0.00152 ± 0.00081 |
| (b) fold swap xf1 - xf0 | W0S +0.02073 ± 0.00089, W0L +0.01551 ± 0.00042, W1S -0.02102 ± 0.00090, W1L -0.01587 ± 0.00093, H0 -0.00030 ± 0.00079 |
| (c) positives per feed, M 5-9 / 10-19 / 20-34 / 35-49 | W0S 1.066 / 1.123 / 1.237 / 1.322 (n 1,266 / 1,872 / 3,930 / 3,954); val M < 50: 1.078 / 1.157 / 1.250 / 1.359 (n 1,043 / 1,505 / 2,030 / 1,315); skeptic (all folds): 1.063 / 1.140 / 1.232 / 1.326. Mean M: W0S 28.1, val M < 50 23.4 |
| (d) resolution (seed-to-seed per-feed sd, xf0 members) | implied SE of a 2-seed member delta: W0S 0.00100, W0L 0.00045, H0 0.00094; cross-user holdout 0.00048; val M < 50 0.00139 |
| rule logic on known arms (`out/ruletest_*.txt`) | A = xf0, B = C = base -> GO (+0.00251 / +0.00376, ratio 1.00); A = xf0, B = xf1, C = base -> KILL (-0.00024) |

The fold-1 controls are not zero (W1S +0.00253 ± 0.00086 = 2.9 SE): base has 25% more users, and the same difference reads +0.00251 ± 0.00048 on the cross-user holdout (2 seeds). It is the data slope, 9 to 14 times smaller than the in-sample margin on the fold-0 witness.

### 1.4 Membership of the witness feeds (recomputed label-free from train.py l.284-318, 430-437, 500-525; counts agree with the six training logs) [M]

| | u20a | u20b | u20c |
|---|---|---|---|
| training feeds | 2,496,456 (624,752 fold-0 feeds dropped) | 3,121,208, of them 624,752 unlabelled | 3,121,208, all labelled |
| cooldown pool (old p_random > 0.5, M >= 50, not holdout) | 107,273 | 107,273 (unlabelled feeds removed) | 134,118 |
| W0S (11,022) | 0 in training | 11,022 unlabelled; 0 in the pool | 11,022 labelled; 0 in the pool |
| W0L (26,883) | 0 in training | 26,883 unlabelled; 0 in the pool | 26,883 labelled; 17,462 in the pool |
| H0 (5,974) | not in training | not in training | not in training |

The only label-dependent factor of an unlabelled feed's target is "has at least one positive" (train.py l.502); 0 witness feeds have no positive. The U20 runs do not use `--prand2`: their cooldown pool comes from the old posterior (`eda/labels_position/out/train_prandom.npz`, zeroed for M < 50), while the witness is defined by the XPOOL v2 posterior (`nn_x/w12/xdet/out/prand_v2.npz`), which also covers short feeds.

## 2. What the numbers say

- **Declared question (inductive, cross-user).** B - A is positive on both witnesses at member level (holdout +0.00061 ± 0.00038, val +0.00099 ± 0.00064) and reaches 43% of C - A on the holdout and about 100% on val [M]. At 2-pool level the holdout reads zero for B (-0.00007 ± 0.00044) against +0.00073 ± 0.00045 for C, and val reads +0.00120 ± 0.00074 for B against +0.00092 ± 0.00078 for C [M]. The two A seeds gain more from pooling (+0.00159 on the cross-user holdout) than the two B seeds or the two C seeds (+0.00090, +0.00089), so B - A vanishes and C - A halves at 2-pool level [M]. With two seeds I cannot tell whether that is a property of the arms (B is more teacher-like: its correlation to the teacher is higher by 0.0064 on every feed set) or of these seed pairs: the A/A null sd of a 2-pool delta is 0.00035 / 0.00027 [M].
- **Transductive question (the analogue of test feeds as a transfer set).** On the feeds whose inputs B trained on, B - A is +0.0006 ± 0.0008 (short) and +0.0010 ± 0.0004 (long) at member level and -0.0004 ± 0.0010 / +0.0003 ± 0.0004 at 2-pool level [M]. Against the cross-user holdout replayed on the CPU (B - A +0.00082 ± 0.00067, n 8,000) the excess is +0.00013 ± 0.00076 on long and -0.00022 ± 0.00106 on short feeds; against the same users' unseen feeds (H0) it is +0.00182 ± 0.00084 on long feeds (@last +0.00122 ± 0.00084) [M]. The two baselines differ by 0.0017 ± 0.0010: on the fold-0 holdout feeds both B - A and the out-of-sample C - A read lower than on the cross-user holdout (GPU files: B - A -0.00111 ± 0.00075 against +0.00061 ± 0.00038; C - A +0.00055 ± 0.00083 against +0.00143 ± 0.00040), and base - xf0 does the same (+0.00127 ± 0.00097 against +0.00251 ± 0.00048, seeds 951, 952). A is common to the first two gaps, so a fluctuation of the two A seeds on these 5,974 feeds would produce both [I]; I have no mechanism; the gaps are 2.0 SE (B - A: -0.00172 ± 0.00084), 1.0 SE (C - A) and 1.1 SE (base - xf0). The label-free mechanism check sides with 'no feed-specific effect': the only training signal B has on a witness feed is the teacher's distribution, and B's extra match to the teacher is the same on long feeds it trained on (+0.00638 ± 0.00004) and on long feeds it never saw (+0.00650 ± 0.00009) [M; the conclusion is I]. B recovers 2.6% / 5.4% of what the labels of those feeds give C (in-sample bound).
- **Short feeds** (39% of test feeds by SYNTHESIS bet 3; not re-measured): no effect is visible (+0.00060 ± 0.00082; re-weighted to the val size mix +0.00011 ± 0.00094; val cross-user M < 50 +0.00022 ± 0.00122) [M]. With SE 0.0008 at 2 seeds the 95% interval on short feeds is about -0.0010 .. +0.0022: a small effect is not excluded.
- **Same-user channel** (fold-0 users' feeds that nobody trained on): holdout -0.00111 ± 0.00075 (GPU) / -0.00086 ± 0.00076 (CPU), val +0.00483 ± 0.00237 (n 852). The two disagree in sign; neither is evidence by the 21:26Z addendum.

## 3. What a KILL closes and what a GO opens (SYNTHESIS bet 5 (a), E_kd_predeclared 20:21Z)

- KILL closes, in one step: val / test feeds as a transfer set, pseudo-feeds with teacher targets,
  candidate swaps, and long function-matching schedules on synthetic views.
- GO (cross-user columns only) opens the more-inputs programme: first a uniform, label-independent sub-feed sampler with recomputed ctx inputs and teacher-only targets where no positive survives (stage 1: new views with logit corr <= 0.98 to the stored view),
  then test feeds as a transfer set, which only public can judge.
- Today neither applies. The read is "2 more seeds".

## 4. Remarks on the rule (I applied it as written)

- The text "otherwise 2 more seeds" does not say how the 4-seed read is decided. A/A null with 4 seeds per arm (70 draws, same-recipe students): sd of the statistic 0.00020 / 0.00018. If the true cross-user effect is +0.0006: P(KILL) 0.17, P(holdout >= +0.0008 and val >= 0) 0.17, P(between the bars again) 0.66; at +0.0004: 0.50 / 0.01 / 0.49; at +0.0008: 0.01 / 0.50 / 0.49 [M under the null model; the stand-in students are full-data d192 students]. The rule for the 4-seed read should be written before seeds 953 / 954 finish.
- The decision statistic is a member delta. Since the 21:26Z block every other rule reads pools. Here the two differ on the holdout (member +0.00061 ± 0.00038, 2-pool -0.00007 ± 0.00044). I report both and decide by the member delta as written.
- The cross-user columns measure the inductive effect. The claim "test feeds as a transfer set" is transductive; table 1.2 is its direct read. A KILL or GO on the cross-user columns alone would close or open it without having looked at it.
- The 20:21Z text names "@last and selected" and no deciding checkpoint. Today both agree; @last is 0.0000026 above the KILL bar, far inside the CPU / GPU and seed noise.

## 5. Method

- Folds: `user_fold` of `wit_common.py` = train.py l.309-310 (uid == 0: feed index % 5; else (uid * 2654435761 % 2^32) % 5); it reproduces `nn_x/w12/recheck/kd_audit/out/xfold_holdout.npy` and `xfold_val.npy` exactly (asserted). Holdout feeds: fold-0 users 5,974, cross-user 24,026. Val: "fold-0 users" = the 852 feeds whose user has a train feed and hashes to fold 0; cross-user = the other 13,571 (unseen users and the user-0 bucket included).
- Declared read (`u20_read.py`): stored `nn_<variant>[@last]_trainonly_val.npy` and `holdout/nn_<variant>[@last]_holdout.npy`; per-feed NDCG with `src/metrics.ndcg_per_feed`; member delta = per-feed mean over the seeds; 2-pool = `feed_rank_z(...)[:, 1]` mean; ratio interval = 1,000 bootstrap draws over feeds.
- Witness sets (`wit_build.py`, label-free): train feeds with v2 posterior > 0.9 outside the 30K holdout, fold 0: W0S (M < 50) and W0L (M >= 50); fold 1: controls W1S and a random 6,000 of the long ones (W1L); H0 = fold-0 holdout feeds. XH (`xh_build.py`) = a random 8,000 of the 24,026 cross-user holdout feeds (seed 20261115).
- Inference path (`wit_common.py`, `wit_score.py`, `xh_score.py`): the arrays that `nn_x/w5/data.SplitData` needs are extracted for the chosen feeds only (np.load mmap, item table compacted) and put into a `SplitData` object without calling its constructor, so `gather / batches / scatter` and `model.Ranker` are the project's own code (imported, not edited). fp32, eval mode, `use_pos=False`, domain token as in training and as train.py scores the holdout and `dump_teacher.py` dumps the teacher ([1 - p, p, 0], p = old posterior, 0 for M < 50). Train labels of the witness rows and holdout labels are read for evaluation only; val / test labels are never model inputs.
- Order of work: witness, validation (a)-(d) and rule tests were finished on 11-14 23:26 UTC, before the first U20 run ended (00:45 UTC). The earlier run also scored one mid-training snapshot of arm A (`u20a_s951@70000`, 00:26 UTC) as a path test; no B or C number existed then. The resumed run re-verified the caches (sample) and re-ran the validation (identical) before scoring any finished U20 checkpoint.

## 6. Limits and what I did NOT check

- Two seeds per arm. Every pool statement is a 2-pool; the final files are 8-16-member pools with a fine-tuned half.
- The witness is an analogue of test feeds, not test: in-period train feeds chosen by a classifier (v2 posterior > 0.9), of users held out by hash; teacher = 3 hard-label members at or below the student's level; students at 80% of users. Test is later in time and its teacher would be the 25-member teacher.
- Step budget: u20a 78,710 steps, u20b and u20c 98,350. Every B - A carries "+25% steps on other data"; no matched-step control exists.
- NOT run for lack of CPU time (machine load 150-160 from 03:00 UTC tripled my scoring time): the witness at @last for u20c (the @last column of C - A), the fold-1 controls W1S / W1L for u20b and u20c, and the val / test domain-token convention ('rl': random-like one-hot instead of the training token) on long feeds and for u20c. The short witness was read under both tokens for A and B (training token [1, 0, 0]: +0.00060 ± 0.00082; val / test token [0, 1, 0]: +0.00070 ± 0.00084; difference of the delta +0.00010 ± 0.00061) [M]. The command in section 7 completes the rest.
- XH is a random 8,000-feed subsample, not the full 24,026 cross-user holdout feeds (full size costs 10 CPU-min per checkpoint at the current load).
- Checkpoint selection uses the 30K holdout, H0 included, for all three arms alike; I did not correct for it. The @last columns are free of it.
- I did not read the fold assignment out of the running processes; membership is recomputed from the train.py code and matches the counts in the six logs (624,752 unlabelled feeds; pools 107,273 / 107,273 / 134,118).
- No GPU replay. CPU fp32 differs from the stored bf16 files by up to 0.0005 for one U20 member on H0; I compare CPU with CPU and GPU file with GPU file only.
- A/A nulls come from full-data kd05w192 / kd05t25w192 students (seeds 951-958), not from fold students; the 4-seed null has only 70 draws.
- I did not run the lead's `nn_x/w12/kd/arm_read8.py` on the U20 arms and did not compare my numbers with the lead's background waiter.
- Rule 1, stated plainly: heavy jobs (12 threads) ran strictly one at a time with CUDA_VISIBLE_DEVICES=""; light read-outs (2 threads, under 2 minutes each) ran next to the one heavy scoring process several times; a few python one-liners that only edited text files of mine (PROGRESS.md, this report; no torch, no data) were started without the variable. Nothing touched a GPU. I stopped four watcher scripts and one scoring pass of my own by exact PID (after reading /proc/<pid>/cmdline) to re-order the queue; no other process was touched.

## 7. Commands

```
cd <repo>
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
PY=.venv/bin/python; D=nn_x/w12/next/u20wit
$PY $D/wit_build.py                                   # witness feed lists + sub-split cache (2-4 min)
$PY $D/wit_score.py tr W0S,H0,W0L,W1S,W1L w5_e10_xf0_s951 w5_e10_xf0_s952 w5_e10_xf0_s953 w5_e10_base_s951 w5_e10_base_s952 w5_e10_base_s953 w5_e10_xf1_s951 w5_e10_xf1_s952 w5_e10_xf1_s953
$PY $D/wit_validate.py > $D/out/wit_validate.txt      # validation (a)-(d)
$PY $D/recheck_cache.py                               # sample re-check of the caches
$PY $D/u20_read.py > $D/out/u20_read.txt              # declared rule, cross-user columns (stored GPU files; no scoring needed)
$PY $D/u20_read_sens.py > $D/out/u20_read_sens.txt    # sensitivity to the cross-user definition
$PY $D/wit_score.py tr W0S,H0,W0L w5_e10_u20a_s951 w5_e10_u20a_s952 w5_e10_u20b_s951 w5_e10_u20b_s952 w5_e10_u20c_s951 w5_e10_u20c_s952
$PY $D/u20_wit_read.py > $D/out/u20_wit_read_tr.txt   # transductive witness
$PY $D/xh_build.py 8000; $PY $D/xh_score.py w5_e10_u20a_s951 w5_e10_u20b_s951 w5_e10_u20a_s952 w5_e10_u20b_s952 w5_e10_u20c_s951 w5_e10_u20c_s952; $PY $D/xh_read.py > $D/out/xh_read.txt
$PY $D/wit_score.py rl W0S w5_e10_u20a_s951 w5_e10_u20a_s952 w5_e10_u20b_s951 w5_e10_u20b_s952; $PY $D/rl_short_read.py > $D/out/rl_short_read.txt   # short witness, val / test token
$PY $D/aa_null_u20_k.py > $D/out/aa_null_u20_k.txt    # A/A null of the statistic, 2 and 4 seeds per arm
# everything in one go, any seed list (idempotent; also completes @last, controls and the rl token):
SEEDS=951,952 setsid nohup bash $D/run_seeds.sh > $D/out/run_seeds_2.log 2>&1 < /dev/null &
SEEDS=951,952,953,954 WAIT=1 setsid nohup bash $D/run_seeds.sh > $D/out/run_seeds_4.log 2>&1 < /dev/null &     # the "2 more seeds" read
```

## 8. Files I created

All under `nn_x/w12/next/u20wit/` (nothing outside it except this report):
- scripts: `wit_common.py`, `wit_build.py`, `wit_score.py`, `wit_validate.py`, `u20_read.py`, `u20_wit_read.py`, `u20_read_sens.py`, `xh_build.py`, `xh_score.py`, `xh_read.py`, `rl_short_read.py`, `aa_null_u20.py`, `aa_null_u20_k.py`, `u20_interim_logs.py`, `recheck_cache.py`, `run_final.sh`, `run_seeds.sh`, `stage1.sh`, `stage2.sh`, `stage2b.sh`, `stage2c.sh`, `stage2d.sh` (the stage scripts are the watchers of this run), `PROGRESS.md`.
- caches: `cache/sets.npz`, `cache/sub/`, `cache/sub_meta.npz`, `cache/sc/` (CPU logits per checkpoint), `cache/subxh/`, `cache/subxh_meta.npz`, `cache/scxh/`, `cache/marks/`, `cache/patch_report_*.py` (helpers that filled numbers into this report).
- outputs in `out/`: `u20_read.txt` (decision; same content as `u20_read_all6.txt`), `u20_wit_read_tr.txt`, `xh_read.txt`, `rl_short_read.txt`, `u20_read_sens.txt`, `aa_null_u20.txt`, `aa_null_u20_k.txt`, `wit_validate.txt` (+ `_rerun`, `_d192`, `_prelim`), `recheck_cache.log`, `lowcorr_feeds.txt`, `u20_membership.txt`, `ruletest_*.txt`, `u20_read_knownanswer_xf0_xf1_base.txt`, `u20_wit_read_knownanswer_xf0_xf1_base.txt`, interim reads (`*_interim_*`, `*_Aonly*`), scoring logs (`score_*.log`, `stage*.log`, `wit_build.log`, `xh_build.log`).
- report: `reports/NEXT_1115/u20wit.md`.
