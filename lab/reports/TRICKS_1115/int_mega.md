# int_mega: what a dramatically larger or better-composed ensemble gives, and what to train next

Date 2026-11-15 (snapshots 21:09Z and 21:43Z). CPU only; read-only on every prediction file; nothing submitted; two candidate files staged in `nn_x/w12/ens_1115/stage/` (not in `subs/`).
Work directory `nn_x/w12/ens_1115/`: scripts `ens_common.py`, `e0_cache.py` .. `e9_report.py`; logs and json in `out/`; z cache in `cache/`.
Tags: [M] measured in this study (file named), [R] record, [W] web source, [I] judgement.
Conventions: every delta is paired per feed over the 14,423 val feeds; "±" is the paired SE; "pmix" is val reweighted to the public / private mix (`src/pmix.py`); "holdout" is the 30,000-feed in-period holdout (out of sample only for members that exclude it; reported, not a veto for popularity-light or staleness arms).
Member score = per-feed z-score `stack_val.feed_rank_z[:, 1]` ((x - mean) / sd inside the feed), the same code path as `nn_x/w12/kd/build_r4.py`; pool = member-equal mean of z; file = 0.5 z(fine-tuned pool) + 0.5 z(train-only pool); per branch 0.8 z(base pool) + 0.2 z(popularity-light pool).
Check of the code path [M, `out/e3_forms.log`]: FD16P20 and W8P20 rebuilt from members match the stored branch files to 9.5e-07 and give final val 0.40271 and 0.40214, the values on record.

## 0. Answer

1. **A larger same-recipe ensemble is worth at most +0.00015 val (about +0.00014 public) over what is already submitted, and that number is now measured three ways** [M]. The law N(n) = N_inf - c / n fits the 16 held-out students, the 16 full-data students and all 32 together with residuals <= 0.00014 on the train-only branch, the fine-tuned branch and the final 0.5 / 0.5 mix (T2). From 16 members: 32 adds +0.00007 ± 0.00001, 64 adds +0.00011 ± 0.00002, 128 adds +0.00013 ± 0.00002, infinitely many +0.00015 ± 0.00003. At file level with the popularity-light pool in place (T3) 16 + 12 -> 32 + 12 is +0.00006 ± 0.00001 and the whole headroom above a 16 + 4 file is +0.00015 ± 0.00003. Paired public SE between near-copy files is 0.0003-0.0007 (0.00034-0.00057 for the staged files, section 4), so no pool size can be read on the public board.
2. **No second arm on disk adds more than extra seeds** [M, T4]. Among 24 arms whose pool level is within 0.001 of the wide students (kd08w192, GRU students, width 224, nine re-cool recipes, kwh6, kwr32, kwd2, kwsb2, kdmix, two generation-2 arms, kx05, kwwd05, kwh30, the d128 students kd05 and kd05t25) the matched "half the pool replaced by the arm" delta averages -0.00007 on val (range -0.00051 .. +0.00026; z mean -0.24, sd 0.73, max +1.11) and -0.00026 on the holdout; none reaches z = 2 on either witness. Cross-arm correlation equals within-arm correlation (0.963-0.980 against 0.9725; every value below 0.970 comes with a lower member level or a holdout loss): under distillation the seed is the only source of diversity. The only input family that decorrelates beyond seed noise is popularity (T12), and it is already in the file at weight 0.2.
3. **The largest honest ensembles available today do not beat FD16P20** [M, T6, T7]. Structure A (32 wide + 36 popularity-light students, 68 members): final val 0.40247 = -0.00024 ± 0.00028 against FD16P20 and +0.00033 ± 0.00036 against W8P20. Structure B (60 members, all trained on all data: 16 wide + 16 hz re-cools + 28 popularity-light): 0.40216 = -0.00056 ± 0.00027 against FD16P20 and +0.00002 ± 0.00044 against W8P20. The hz re-cools lower the fine-tuned branch again (FD16 + FDXHZ16 against FD16, same four popularity-light students: file -0.00034 ± 0.00025, fine-tuned branch -0.00038 ± 0.00024; the shape of the three member sets on record).
4. **FD16P20's val level is a favourable draw, and the draw does not transfer** [M, T10]. Its four popularity-light students put it above 472 of the 495 possible "4 of the 12" files (mean 0.40249, sd 0.00013). Across random halves of the val users the draw excess has correlation -0.16 (val), -0.03 (public mix), -0.005 (private mix): reliability <= 0. Against the expected FD16P20-type file A reads -0.00002 ± 0.00024 and B -0.00033 ± 0.00022. So A is level with the recipe, B is below it, and neither is a gain.
5. **Learned combiners are worth zero nested** [M, T8]. Ten forms (Caruana greedy with replacement, non-negative least squares, ridge; member or arm columns; all members or all-data members only): nested deltas against B from -0.00083 ± 0.00042 to +0.00020 ± 0.00034, all below FD16P20; in-sample the same forms read up to +0.00177 above B. Twenty scan forms (arm-balanced weights, d128 / hard-label admixture, rank / softmax / trimmed / median pooling) are all inside ±2 SE of their structure except one negative; the best one (rank pooling, +0.00032 on B and +0.00022 on A) was put to a pre-written replication read on four disjoint member sets and did not replicate (pooled -0.00007, z -0.5; T9).
6. **What to train next for the ensemble: nothing pays for level** [I from 2]. Sixteen more full-data wide students cost about 70 GPU-hours for +0.00006 ± 0.00001; cooldown re-runs of existing snapshots correlate 0.990 with their root and cap at +0.00007 on the train-only branch; fine-tunes of the 76 held-out members that lack one would move a held-out file by -0.0003 .. +0.0003 (all inside noise, law value <= +0.0001). A second arm would need a cross-arm correlation near 0.945 at unchanged member level to add +0.0005 (frontier regression, T4); the lowest value on disk among wide-level arms is 0.963, and the popularity-light students sit at 0.961 with a pool level 0.001 lower.
7. **Staged** (section 4): `cand_ENSAkd_stage.parquet` (A) and `cand_ENSBkd_stage.parquet` (B) with branch files, member lists, sha256 and read-outs. No third or fourth file: no form beats B by more than 2 SE nested. Neither file is recommended for a slot as a level bet; A is a lower-variance copy of the recipe (not all-data), B is all-data and expected below FD16P20.

## 1. Inventory [M]

Source: `e0_cache.py` -> `out/inventory.json` (per member: path used, number of identical copies, file time, NDCG), `e1_inventory.py` -> `out/inventory_arms.json` / `.md`, `e1b_outside.py` -> `out/inventory_outside.md`.
On disk at 21:43Z: 1,223 member names over `preds/`, `nn_x/w5/preds/`, `nn_x/abl/preds/` and their `holdout/` folders. 633 are in the cache: every 10-epoch-class member that has a train-only val AND a train-only test file (selected checkpoint), plus the last-step twins of the re-cool arms and of the 8 held-out hz re-cools that were fine-tuned at the last step. 132 members have the fine-tuned pair (n2cpv3s3 val_oof + test). 573 have a holdout file. Every file that exists in two directories was compared: 0 differing copies. Full-data flag = `args.holdout_in_train` in the member's training-log json: 60 members.
kwsb2 x4 finished at 21:25Z and is included (second cache build 21:43Z); kwaux1 and u20 seeds 953-954 were still training and are not on disk.

| class | arm | members | seeds | all data | TO val + test | FT val + test | holdout file | TO val member | TO val arm pool | FT val arm pool | holdout arm pool (out of sample) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wide students (the recipe) | w5_e10_kd05t25w192 | 8 | 951-958 | no | 8 | 8 | 8 | 0.39963 | 0.40154 | 0.40107 | 0.32409 |
| wide students (the recipe) | w5_e10_kd05w192 | 8 | 951-958 | no | 8 | 8 | 8 | 0.39902 | 0.40110 | 0.39991 | 0.32430 |
| wide students (the recipe) | w5_fd10_kd05t25w192 | 8 | 951-958 | yes | 8 | 8 | 0 | 0.39951 | 0.40160 | 0.40122 |  |
| wide students (the recipe) | w5_fd10_kd05w192 | 8 | 951-958 | yes | 8 | 8 | 0 | 0.39940 | 0.40113 | 0.40094 |  |
| all-data re-cools | w5_fdxhz_kd05t25w192 | 8 | 951-958 | yes | 8 | 8 | 0 | 0.39979 | 0.40206 | 0.40157 |  |
| all-data re-cools | w5_fdxhz_kd05w192 | 8 | 951-958 | yes | 8 | 8 | 0 | 0.39960 | 0.40144 | 0.40058 |  |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hz | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39949 | 0.40104 / 0.40146 |  | 0.32367 / 0.32356 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hz2 | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39967 | 0.40112 / 0.40178 |  | 0.32153 / 0.32137 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hz2@last | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39976 | 0.40137 / 0.40169 |  | 0.32143 / 0.32133 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hz2nF | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39966 | 0.40110 / 0.40141 |  | 0.32144 / 0.32156 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hz2nF@last | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39987 | 0.40131 / 0.40172 |  | 0.32139 / 0.32125 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hz@last | 16 | 951-958 (K and L) | no | 16 | 8 | 16 | 0.39950 | 0.40108 / 0.40154 | 0.40089 / 0.40020 | 0.32369 / 0.32341 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hznFS | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39963 | 0.40116 / 0.40142 |  | 0.32390 / 0.32366 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hznFS@last | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39962 | 0.40157 / 0.40173 |  | 0.32384 / 0.32339 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hzt | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39954 | 0.40127 / 0.40117 |  | 0.32346 / 0.32329 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_hzt@last | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39961 | 0.40129 / 0.40141 |  | 0.32327 / 0.32326 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_kw08 | 1 | 953 | no | 1 | 0 | 1 | 0.39955 | 0.39955 |  | 0.32342 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_kw08@last | 1 | 953 | no | 1 | 0 | 1 | 0.39955 | 0.39955 |  | 0.32342 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_ref | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39903 | 0.40081 / 0.40113 |  | 0.32406 / 0.32417 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_ref2 | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39906 | 0.40082 / 0.40134 |  | 0.32423 / 0.32421 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_ref2@last | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39928 | 0.40131 / 0.40140 |  | 0.32400 / 0.32398 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_ref@last | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39918 | 0.40118 / 0.40102 |  | 0.32418 / 0.32412 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_th | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39926 | 0.40113 / 0.40082 |  | 0.32367 / 0.32397 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_th@last | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39943 | 0.40152 / 0.40116 |  | 0.32365 / 0.32393 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_xd30 | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39894 | 0.40025 / 0.40049 |  | 0.32344 / 0.32306 |
| held-out re-cools of the 16 wide snapshots | w5_{k,l}cd_xd30@last | 16 | 951-958 (K and L) | no | 16 | 0 | 16 | 0.39897 | 0.40064 / 0.40077 |  | 0.32354 / 0.32296 |
| popularity-light students | abl_e10_pop1kd05t25 | 4 | 951-954 | no | 4 | 4 | 4 | 0.39821 | 0.39943 | 0.40033 | 0.31715 |
| popularity-light students | abl_e10_pop1kd05w192 | 4 | 951-954 | no | 4 | 4 | 4 | 0.39859 | 0.40000 | 0.40099 | 0.31772 |
| popularity-light students | abl_fd10_pfkd05w192 | 8 | 951-958 | yes | 8 | 8 | 0 | 0.39787 | 0.39955 | 0.40044 |  |
| popularity-light students | abl_fd10_pop1kd05t25 | 4 | 951-954 | yes | 4 | 4 | 0 | 0.39829 | 0.39969 | 0.40047 |  |
| popularity-light students | abl_fd10_pop1kd05w192 | 12 | 951-962 | yes | 12 | 12 | 0 | 0.39848 | 0.40023 | 0.40114 |  |
| popularity-light students | abl_fd10_pop2kd05w192 | 4 | 951-954 | yes | 4 | 4 | 0 | 0.39804 | 0.39990 | 0.40019 |  |
| other d192 / d224 / d256 student arms | w5_e10_kd03g2w192 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39762 | 0.40093 |  | 0.32393 |
| other d192 / d224 / d256 student arms | w5_e10_kd05g2w192 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39914 | 0.40080 |  | 0.32410 |
| other d192 / d224 / d256 student arms | w5_e10_kd05t25w192b3 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39936 | 0.40026 |  | 0.32348 |
| other d192 / d224 / d256 student arms | w5_e10_kd05t25w256 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39912 | 0.40014 |  | 0.32310 |
| other d192 / d224 / d256 student arms | w5_e10_kd05w192gru | 4 | 951-954 | no | 4 | 0 | 4 | 0.39973 | 0.40134 |  | 0.32390 |
| other d192 / d224 / d256 student arms | w5_e10_kd05w224 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39935 | 0.40160 |  | 0.32424 |
| other d192 / d224 / d256 student arms | w5_e10_kd08w192 | 8 | 951-958 | no | 8 | 0 | 8 | 0.40028 | 0.40061 |  | 0.32401 |
| other d192 / d224 / d256 student arms | w5_e10_kdmixw192 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39900 | 0.40104 |  | 0.32421 |
| other d192 / d224 / d256 student arms | w5_e10_kw256d2 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39837 | 0.39970 |  | 0.32333 |
| other d192 / d224 / d256 student arms | w5_e10_kwT07 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39885 | 0.40023 |  | 0.32362 |
| other d192 / d224 / d256 student arms | w5_e10_kwT15 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39936 | 0.40038 |  | 0.32359 |
| other d192 / d224 / d256 student arms | w5_e10_kw_drop0 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39854 | 0.40011 |  | 0.32394 |
| other d192 / d224 / d256 student arms | w5_e10_kw_e20 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39834 | 0.40036 |  | 0.32373 |
| other d192 / d224 / d256 student arms | w5_e10_kw_ff4 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39806 | 0.39949 |  | 0.32330 |
| other d192 / d224 / d256 student arms | w5_e10_kw_hl2 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39932 | 0.40005 |  | 0.32292 |
| other d192 / d224 / d256 student arms | w5_e10_kw_id | 2 | 951,952 | no | 2 | 0 | 2 | 0.39343 | 0.39527 |  | 0.31938 |
| other d192 / d224 / d256 student arms | w5_e10_kwaux0 | 8 | 951-958 | no | 8 | 0 | 8 | 0.39742 | 0.39976 |  | 0.32302 |
| other d192 / d224 / d256 student arms | w5_e10_kwd2 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39918 | 0.40066 |  | 0.32362 |
| other d192 / d224 / d256 student arms | w5_e10_kwh30 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39899 | 0.40035 |  | 0.32342 |
| other d192 / d224 / d256 student arms | w5_e10_kwh6 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39877 | 0.40081 |  | 0.32464 |
| other d192 / d224 / d256 student arms | w5_e10_kwr32 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39902 | 0.40036 |  | 0.32365 |
| other d192 / d224 / d256 student arms | w5_e10_kwsb2 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39913 | 0.40081 |  | 0.32365 |
| other d192 / d224 / d256 student arms | w5_e10_kwsub2 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39938 | 0.40022 |  | 0.32347 |
| other d192 / d224 / d256 student arms | w5_e10_kwsub5 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39939 | 0.40001 |  | 0.32363 |
| other d192 / d224 / d256 student arms | w5_e10_kwwd05 | 8 | 951-958 | no | 8 | 0 | 8 | 0.39958 | 0.40067 |  | 0.32367 |
| other d192 / d224 / d256 student arms | w5_e10_kwwd10 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39894 | 0.39965 |  | 0.32317 |
| other d192 / d224 / d256 student arms | w5_e10_u20a | 2 | 951,952 | no | 2 | 0 | 2 | 0.39667 | 0.39797 |  | 0.32259 |
| other d192 / d224 / d256 student arms | w5_e10_u20b | 2 | 951,952 | no | 2 | 0 | 2 | 0.39788 | 0.39955 |  | 0.32242 |
| other d192 / d224 / d256 student arms | w5_e10_u20c | 2 | 951,952 | no | 2 | 0 | 2 | 0.39769 | 0.39895 |  | 0.32324 |
| d128 students | w5_e10_kd03 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39806 | 0.39977 |  | 0.32342 |
| d128 students | w5_e10_kd05 | 4 | 951-954 | no | 4 | 4 | 4 | 0.39911 | 0.40085 | 0.40049 | 0.32349 |
| d128 students | w5_e10_kd05t25 | 4 | 951-954 | no | 4 | 4 | 4 | 0.39957 | 0.40095 | 0.40080 | 0.32317 |
| d128 students | w5_e10_kd08 | 4 | 951-954 | no | 4 | 4 | 4 | 0.39928 | 0.40004 | 0.39996 | 0.32377 |
| d128 students | w5_e10_kd08t25 | 3 | 951-953 | no | 3 | 0 | 3 | 0.40034 | 0.40062 |  | 0.32295 |
| d128 students | w5_e10_kd10 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39883 | 0.39906 |  | 0.32291 |
| d128 students | w5_e10_kx05 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39916 | 0.40079 |  | 0.32347 |
| d128 students | w5_e10_kx08 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39938 | 0.40008 |  | 0.32321 |
| hard-label members | abl_e10_blocks3 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39691 | 0.39958 |  | 0.32171 |
| hard-label members | abl_e10_latedrop | 2 | 951,952 | no | 2 | 0 | 2 | 0.39667 | 0.39870 |  | 0.32241 |
| hard-label members | abl_e10_pop1 | 8 | 951-958 | no | 8 | 8 | 8 | 0.39467 | 0.39841 | 0.39863 | 0.31717 |
| hard-label members | abl_e10_pop2 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39472 | 0.39760 |  | 0.31625 |
| hard-label members | abl_e10_retr32 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39643 | 0.39918 |  | 0.32170 |
| hard-label members | abl_e10_retr32u | 2 | 951,952 | no | 2 | 0 | 2 | 0.39635 | 0.39832 |  | 0.32205 |
| hard-label members | abl_e10_xdrop_pop | 4 | 951-954 | no | 4 | 0 | 4 | 0.39403 | 0.39727 |  | 0.31476 |
| hard-label members | abl_e10_xdrop_pre2 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39584 | 0.39776 |  | 0.32168 |
| hard-label members | abl_e10_xrel_more | 2 | 951,952 | no | 2 | 0 | 2 | 0.39472 | 0.39662 |  | 0.32159 |
| hard-label members | r6_cd_m02 | 7 | 101-107 | no | 7 | 7 | 7 | 0.39667 | 0.40061 | 0.40053 | 0.32332 |
| hard-label members | r6_cd_m02_p70k | 7 | 101-107 | no | 7 | 7 | 7 | 0.39621 | 0.40024 | 0.39985 | 0.32315 |
| hard-label members | v14 | 5 | 92-96 | no | 5 | 0 | 5 | 0.39275 | 0.39702 |  | 0.32150 |
| hard-label members | v17_pre2s | 7 | 101-107 | no | 7 | 0 | 7 | 0.39460 | 0.39867 |  | 0.32291 |
| hard-label members | v22_pre2s_lr1 | 6 | 401-406 | no | 6 | 0 | 6 | 0.39382 | 0.39820 |  | 0.32213 |
| hard-label members | v23_gru_pre2s | 6 | 501-506 | no | 6 | 6 | 6 | 0.39481 | 0.39848 | 0.39906 | 0.32254 |
| hard-label members | v26_d192_pre2s | 7 | 701-707 | no | 7 | 0 | 7 | 0.39371 | 0.39829 |  | 0.32250 |
| hard-label members | w5_e10_ap1 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39710 | 0.39897 |  | 0.32216 |
| hard-label members | w5_e10_base | 6 | 951-956 | no | 6 | 0 | 6 | 0.39648 | 0.39993 |  | 0.32347 |
| hard-label members | w5_e10_bs4 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39400 | 0.39630 |  | 0.32000 |
| hard-label members | w5_e10_bs4lr4 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39430 | 0.39641 |  | 0.32064 |
| hard-label members | w5_e10_e15 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39607 | 0.39808 |  | 0.32184 |
| hard-label members | w5_e10_e20 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39576 | 0.39942 |  | 0.32305 |
| hard-label members | w5_e10_e30 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39507 | 0.39739 |  | 0.32256 |
| hard-label members | w5_e10_pseudo | 2 | 951,952 | no | 2 | 0 | 2 | 0.39769 | 0.39909 |  | 0.32215 |
| hard-label members | w5_e10_pseudo_xpre2 | 2 | 951,952 | no | 2 | 0 | 2 | 0.39578 | 0.39745 |  | 0.32205 |
| hard-label members | w5_e10_trunc | 4 | 951-954 | no | 4 | 0 | 4 | 0.39607 | 0.39853 |  | 0.32241 |
| hard-label members | w5_e10_trunc2 | 4 | 951-954 | no | 4 | 0 | 4 | 0.39618 | 0.39959 |  | 0.32282 |
| hard-label members | w5_e10_xf0 | 3 | 951-953 | no | 3 | 0 | 3 | 0.39369 | 0.39710 |  | 0.32091 |
| hard-label members | w5_e10_xf1 | 3 | 951-953 | no | 3 | 0 | 3 | 0.39451 | 0.39663 |  | 0.32118 |
| hard-label members | w5_e10_xf2 | 3 | 951-953 | no | 3 | 0 | 3 | 0.39401 | 0.39657 |  | 0.32100 |
| hard-label members | w5_e10_xf3 | 3 | 951-953 | no | 3 | 0 | 3 | 0.39402 | 0.39751 |  | 0.32142 |
| hard-label members | w5_e10_xf4 | 3 | 951-953 | no | 3 | 0 | 3 | 0.39450 | 0.39863 |  | 0.32095 |

Train-only val pool values with two numbers are the K lineage (kd05w192 snapshots) / the L lineage (kd05t25w192 snapshots). The four files of a fine-tuned member are all present for the 132 members counted in "FT val + test"; no member has a fine-tuned val file without the test file.

Members on disk outside the cache (not used; names and file counts):

| family | members | TO val | TO test | FT val | FT test | holdout |
|---|---|---|---|---|---|---|
| 2-epoch ablation members (seeds 23, 24): abl_base | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_blocks3 | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_kv64 | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_lnorm_batch | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_no_aux | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_no_domain | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_no_retr | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_no_set | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_no_sim | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_pop1 | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_pop2 | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_pre2noid | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_retr32 | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_retr32u | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_retr_cat | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_smoke_kv64 | 1 | 1 | 0 | 0 | 0 | 1 |
| 2-epoch ablation members (seeds 23, 24): abl_smoke_pre2noid | 1 | 1 | 0 | 0 | 0 | 1 |
| 2-epoch ablation members (seeds 23, 24): abl_time | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_xdrop_ctxhn_hs | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_xdrop_ctxnd | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_xdrop_pop | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_xdrop_pre2 | 2 | 2 | 0 | 0 | 0 | 2 |
| 2-epoch ablation members (seeds 23, 24): abl_xrel_more | 2 | 2 | 0 | 0 | 0 | 2 |
| @last twin of a from-scratch run (same run, last-step checkpoint) | 435 | 435 | 391 | 0 | 0 | 372 |
| other: ens_stage1 | 1 | 1 | 1 | 0 | 0 | 0 |
| other: final2 | 1 | 1 | 1 | 0 | 0 | 0 |
| other: final5 | 1 | 1 | 1 | 0 | 0 | 0 |
| other: v13_lr1_mix1 | 1 | 1 | 1 | 0 | 0 | 1 |
| other: v15_cd_all | 2 | 2 | 2 | 0 | 0 | 2 |
| other: v15_cd_all_e10 | 1 | 1 | 1 | 0 | 0 | 1 |
| other: v15_cd_allprew | 3 | 3 | 3 | 0 | 0 | 3 |
| smoke runs | 3 | 3 | 3 | 0 | 0 | 0 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_aa0 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_ap1 | 7 | 7 | 7 | 0 | 0 | 7 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_auxf | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_cbs025 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_cbs05 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_cbs2 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_fpcd | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_hzcd | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_m01 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_m02ref | 7 | 7 | 7 | 0 | 0 | 7 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_m04 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_rdrop | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_sam | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_truncp | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_truncp2 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_xp2 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_xp2m05 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_* hard-label cooldown screens on the F7 roots: w5_cd_xp2m10 | 4 | 4 | 4 | 0 | 0 | 4 |
| w5_cd_valcd (cooldown on val folds, 4 roots x 5 folds) | 20 | 20 | 20 | 0 | 0 | 20 |

The `w5_cd_valcd` members were cooled on val folds; their val files are not out of sample and must not enter any pool.

## 2. Pool math [M]

### 2.1 Pool-size law for the wide students

`e2_law.py` -> `out/law.json`, `out/e2_law.log`. For every n up to 200 distinct random subsets (all subsets when fewer); the subset mean of the per-feed NDCG estimates E[N(n)] over seeds. Fit by least squares over n = 1..N. SE of c = feed SE (the fit is linear in the per-n means) combined with a member jackknife; the forecast interval is that SE times (1/16 - 1/K).

| set | branch | N_inf | c | SE of c (feeds) | SE of c (member jackknife) | max residual | 16 -> 32 | 16 -> 64 | 16 -> 128 | 16 -> infinity | realised N(all) minus mean half-size pool (direct) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W16 | train-only | 0.40167 | 0.00239 | 0.00036 | 0.00018 | 0.00008 | +0.00007 ± 0.00001 | +0.00011 ± 0.00002 | +0.00013 ± 0.00002 | +0.00015 ± 0.00003 | +0.00004 ± 0.00017 |
| W16 | fine-tuned | 0.40094 | 0.00196 | 0.00035 | 0.00016 | 0.00012 | +0.00006 ± 0.00001 | +0.00009 ± 0.00002 | +0.00011 ± 0.00002 | +0.00012 ± 0.00002 | -0.00003 ± 0.00017 |
| W16 | final 0.5 / 0.5 | 0.40176 | 0.00214 | 0.00035 | 0.00017 | 0.00005 | +0.00007 ± 0.00001 | +0.00010 ± 0.00002 | +0.00012 ± 0.00002 | +0.00013 ± 0.00002 | +0.00008 ± 0.00017 |
| W16 | train-only, holdout | 0.32464 | 0.00215 | 0.00020 | 0.00009 | 0.00009 | +0.00007 ± 0.00001 | +0.00010 ± 0.00001 | +0.00012 ± 0.00001 | +0.00013 ± 0.00001 | +0.00025 ± 0.00009 |
| FD16 | train-only | 0.40169 | 0.00226 | 0.00035 | 0.00022 | 0.00008 | +0.00007 ± 0.00001 | +0.00011 ± 0.00002 | +0.00012 ± 0.00002 | +0.00014 ± 0.00003 | +0.00024 ± 0.00017 |
| FD16 | fine-tuned | 0.40144 | 0.00237 | 0.00035 | 0.00017 | 0.00011 | +0.00007 ± 0.00001 | +0.00011 ± 0.00002 | +0.00013 ± 0.00002 | +0.00015 ± 0.00002 | +0.00004 ± 0.00016 |
| FD16 | final 0.5 / 0.5 | 0.40208 | 0.00245 | 0.00035 | 0.00019 | 0.00011 | +0.00008 ± 0.00001 | +0.00011 ± 0.00002 | +0.00013 ± 0.00002 | +0.00015 ± 0.00002 | +0.00010 ± 0.00017 |
| W32 | train-only | 0.40160 | 0.00219 | 0.00037 | 0.00017 | 0.00014 | +0.00007 ± 0.00001 | +0.00010 ± 0.00002 | +0.00012 ± 0.00002 | +0.00014 ± 0.00003 | -0.00010 ± 0.00014 |
| W32 | fine-tuned | 0.40129 | 0.00233 | 0.00037 | 0.00013 | 0.00011 | +0.00007 ± 0.00001 | +0.00011 ± 0.00002 | +0.00013 ± 0.00002 | +0.00015 ± 0.00002 | +0.00021 ± 0.00014 |
| W32 | final 0.5 / 0.5 | 0.40195 | 0.00234 | 0.00037 | 0.00013 | 0.00007 | +0.00007 ± 0.00001 | +0.00011 ± 0.00002 | +0.00013 ± 0.00002 | +0.00015 ± 0.00002 | +0.00004 ± 0.00014 |

Curves E[N(n)] (final val): W16: n=1 0.39960, n=2 0.40070, n=4 0.40125, n=8 0.40150, n=16 0.40158 | FD16: n=1 0.39973, n=2 0.40077, n=4 0.40137, n=8 0.40176, n=16 0.40187 | W32: n=1 0.39966, n=2 0.40074, n=4 0.40133, n=8 0.40167, n=16 0.40184, n=32 0.40189

Reading.
- The three sets give the same law: c between 0.0020 and 0.0025, N_inf (final) 0.40176 / 0.40208 / 0.40195. The held-out and the full-data students are the same recipe to within the SE (same-seed pairs across the two sets correlate 0.9732, different-seed pairs 0.9726, pairs inside a set 0.9725 / 0.9737: the full-data twin of a seed is an independent member) [M, `e2d_misc.py` -> `out/misc_corr.json`].
- Out-of-range check: a fit on n <= N/2 predicts the realised pool of N with error -0.00015 .. +0.00017. Direct check: the realised 32-pool minus the mean 16-pool is -0.00010 ± 0.00014 (train-only), +0.00021 ± 0.00014 (fine-tuned), +0.00004 ± 0.00014 (final); the law says +0.00007.
- Under the public / private mix the constant is the same (final: 0.00251 / 0.00239 for W32, `out/law.json` keys `all_public`, `all_private`).
- One realised 16-member file carries val draw noise: sd 0.00020 over 16-of-32 draws at file level (T10). That noise is larger than every gain in this table, and it does not transfer to new feeds (section 2.5).

### 2.2 The same law at file level, both pools

`e2c_file_law.py` -> `out/file_law.json`. All-data members: nb of the 16 full-data wide students as base, np of the 12 full-data wide pop1 students at weight 0.2, 40 random draws per cell, final val.

| weights | a | cb | cp | max residual | 16+4 -> 16+12 | 16+12 -> 32+12 | 16+12 -> 16+24 | 16+12 -> 64+48 | 16+12 -> infinity | 16+4 -> infinity | one more base student at 16 | one more pop student at 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 0.40254 | 0.00177 ± 0.00027 | 0.00014 ± 0.00005 | 0.00019 | +0.00002 ± 0.00001 | +0.00006 ± 0.00001 | +0.00001 ± 0.00000 | +0.00009 ± 0.00001 | +0.00012 ± 0.00002 | +0.00015 ± 0.00003 | +0.000007 | +0.000001 |
| pmix public | 0.39965 | 0.00177 ± 0.00033 | 0.00011 ± 0.00006 | 0.00017 | +0.00002 ± 0.00001 | +0.00006 ± 0.00001 | +0.00000 ± 0.00000 | +0.00009 ± 0.00002 | +0.00012 ± 0.00002 | +0.00014 ± 0.00003 | +0.000007 | +0.000001 |
| pmix private | 0.40525 | 0.00167 ± 0.00040 | 0.00008 ± 0.00007 | 0.00022 | +0.00001 ± 0.00001 | +0.00005 ± 0.00001 | +0.00000 ± 0.00000 | +0.00008 ± 0.00002 | +0.00011 ± 0.00003 | +0.00012 ± 0.00004 | +0.000007 | +0.000001 |

Reading. The popularity-light pool is saturated at once (cp = 0.00014: one student already carries the admixture; 4 -> 12 students is worth +0.00002). The base pool carries cb = 0.00177, lower than the 0.0021-0.0025 of a base-only file because the admixture already removes part of the noise. Total headroom above a 16 + 4 file: +0.00015 ± 0.00003 (val), +0.00014 ± 0.00003 (public mix), +0.00012 ± 0.00004 (private mix).

### 2.3 A second arm at equal size

`e2_arms.py` -> `out/arms.json`, `out/arms_late.json` (kwsb2), `out/e2_arms.log`. For every arm with at least 4 members: correlation of per-feed z inside the arm and against the 16 held-out wide students; pool level of h members against h wide students; "replace" = pool(h arm + h wide) minus pool(2h wide) with disjoint seeds (re-cool arms: disjoint snapshot roots, and the removed wide students are the ORIGINAL runs of the arm's roots, so the contrast is the recipe at equal independence); "add" = all arm members added to W16. 60 random draws per arm. h = 8 for arms with at least 8 members, else 4. Re-cool arms are read at the last step (the decision checkpoint of the lead's rule); the selected-checkpoint rows are in the json.

| arm | n | member val \| hold | corr within arm | corr arm x wide (different seed / root) | same seed / root | h | level vs h wide | replace half of a 2h wide pool (val) | (holdout) | pmix public / private | add all to W16 (val \| hold) | within 0.001 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w5_e10_kd08w192 | 8 | 0.40028 \| 0.32337 | 0.9900 | 0.9797 | 0.9807 | 8 | -0.00079 ± 0.00040 \| -0.00033 ± 0.00023 | -0.00029 ± 0.00027 | -0.00025 ± 0.00015 | -0.00056 / -0.00064 | -0.00027 ± 0.00024 \| -0.00009 ± 0.00014 | yes |
| w5_e10_kd05w192gru | 4 | 0.39973 \| 0.32268 | 0.9728 | 0.9710 | 0.9720 | 4 | +0.00023 ± 0.00049 \| -0.00017 ± 0.00027 | -0.00001 ± 0.00028 | -0.00023 ± 0.00015 | +0.00028 / +0.00050 | +0.00010 ± 0.00025 \| -0.00006 ± 0.00012 | yes |
| w5_e10_kd05w224 | 4 | 0.39935 \| 0.32270 | 0.9693 | 0.9704 | 0.9718 | 4 | +0.00050 ± 0.00045 \| +0.00016 ± 0.00026 | +0.00004 ± 0.00024 | +0.00021 ± 0.00014 | -0.00028 / -0.00038 | +0.00007 ± 0.00021 \| -0.00006 ± 0.00012 | yes |
| RC:ref@last | 16 | 0.39918 \| 0.32232 | 0.9723 | 0.9723 | 0.9904 | 8 | -0.00021 ± 0.00018 \| -0.00020 ± 0.00011 | +0.00000 ± 0.00017 | -0.00018 ± 0.00010 | -0.00002 / +0.00001 | +0.00004 ± 0.00020 \| -0.00013 ± 0.00011 | yes |
| RC:ref2@last | 16 | 0.39928 \| 0.32227 | 0.9720 | 0.9721 | 0.9904 | 8 | +0.00000 ± 0.00019 \| -0.00022 ± 0.00010 | +0.00020 ± 0.00018 | -0.00020 ± 0.00010 | +0.00010 / +0.00007 | +0.00011 ± 0.00021 \| -0.00020 ± 0.00011 | yes |
| RC:hz@last | 16 | 0.39950 \| 0.32180 | 0.9733 | 0.9715 | 0.9886 | 8 | -0.00005 ± 0.00031 \| -0.00071 ± 0.00019 | +0.00008 ± 0.00024 | -0.00024 ± 0.00015 | +0.00014 / +0.00022 | +0.00009 ± 0.00026 \| -0.00014 ± 0.00016 | yes |
| RC:th@last | 16 | 0.39943 \| 0.32206 | 0.9727 | 0.9722 | 0.9900 | 8 | -0.00015 ± 0.00024 \| -0.00055 ± 0.00016 | +0.00000 ± 0.00020 | -0.00042 ± 0.00013 | +0.00010 / +0.00021 | -0.00008 ± 0.00023 \| -0.00044 ± 0.00015 | yes |
| RC:xd30@last | 16 | 0.39897 \| 0.32156 | 0.9737 | 0.9694 | 0.9859 | 8 | -0.00073 ± 0.00039 \| -0.00106 ± 0.00019 | -0.00028 ± 0.00028 | -0.00034 ± 0.00014 | -0.00017 / -0.00008 | -0.00039 ± 0.00031 \| -0.00031 ± 0.00016 | yes |
| RC:hzt@last | 16 | 0.39961 \| 0.32160 | 0.9736 | 0.9713 | 0.9881 | 8 | -0.00007 ± 0.00035 \| -0.00106 ± 0.00021 | +0.00008 ± 0.00026 | -0.00040 ± 0.00016 | +0.00018 / +0.00029 | +0.00008 ± 0.00029 \| -0.00030 ± 0.00018 | yes |
| RC:hznFS@last | 16 | 0.39962 \| 0.32188 | 0.9732 | 0.9710 | 0.9880 | 8 | +0.00010 ± 0.00033 \| -0.00061 ± 0.00019 | +0.00018 ± 0.00025 | -0.00019 ± 0.00015 | +0.00026 / +0.00034 | +0.00025 ± 0.00027 \| -0.00014 ± 0.00017 | yes |
| RC:hz2@last | 16 | 0.39976 \| 0.31967 | 0.9739 | 0.9680 | 0.9839 | 8 | +0.00008 ± 0.00049 \| -0.00292 ± 0.00035 | +0.00017 ± 0.00034 | -0.00118 ± 0.00023 | +0.00044 / +0.00068 | +0.00003 ± 0.00037 \| -0.00114 ± 0.00025 | yes |
| RC:hz2nF@last | 16 | 0.39987 \| 0.31956 | 0.9737 | 0.9668 | 0.9828 | 8 | +0.00020 ± 0.00049 \| -0.00294 ± 0.00035 | +0.00008 ± 0.00033 | -0.00117 ± 0.00023 | +0.00026 / +0.00040 | -0.00020 ± 0.00036 \| -0.00120 ± 0.00025 | yes |
| w5_e10_kwh6 | 4 | 0.39877 \| 0.32273 | 0.9710 | 0.9714 | 0.9731 | 4 | -0.00030 ± 0.00044 \| +0.00057 ± 0.00026 | -0.00017 ± 0.00025 | +0.00028 ± 0.00014 | -0.00034 / -0.00056 | -0.00001 ± 0.00020 \| -0.00004 ± 0.00012 | yes |
| w5_e10_kwr32 | 4 | 0.39902 \| 0.32241 | 0.9705 | 0.9706 | 0.9720 | 4 | -0.00075 ± 0.00049 \| -0.00042 ± 0.00028 | -0.00012 ± 0.00027 | -0.00017 ± 0.00016 | -0.00010 / -0.00027 | +0.00017 ± 0.00023 \| -0.00016 ± 0.00012 | yes |
| w5_e10_kwd2 | 4 | 0.39918 \| 0.32238 | 0.9773 | 0.9739 | 0.9757 | 4 | -0.00044 ± 0.00044 \| -0.00046 ± 0.00025 | -0.00051 ± 0.00025 | -0.00032 ± 0.00014 | -0.00055 / -0.00053 | +0.00017 ± 0.00021 \| -0.00020 ± 0.00011 | yes |
| w5_e10_kwsb2 | 4 | 0.39913 \| 0.32232 | 0.9713 | 0.9704 | 0.9741 | 4 | -0.00029 ± 0.00046 \| -0.00042 ± 0.00027 | -0.00016 ± 0.00026 | -0.00018 ± 0.00014 | -0.00013 / -0.00014 | -0.00017 ± 0.00021 \| -0.00012 ± 0.00012 | yes |
| w5_e10_kdmixw192 | 4 | 0.39900 \| 0.32236 | 0.9691 | 0.9706 | 0.9749 | 4 | -0.00006 ± 0.00044 \| +0.00013 ± 0.00025 | +0.00026 ± 0.00024 | +0.00009 ± 0.00013 | +0.00033 / +0.00035 | +0.00033 ± 0.00022 \| -0.00013 ± 0.00012 | yes |
| w5_e10_kd05g2w192 | 4 | 0.39914 \| 0.32252 | 0.9702 | 0.9705 | 0.9755 | 4 | -0.00030 ± 0.00044 \| +0.00003 ± 0.00026 | -0.00014 ± 0.00025 | +0.00005 ± 0.00014 | -0.00023 / -0.00027 | +0.00012 ± 0.00020 \| -0.00015 ± 0.00012 | yes |
| w5_e10_kd03g2w192 | 4 | 0.39762 \| 0.32136 | 0.9548 | 0.9630 | 0.9660 | 4 | -0.00017 ± 0.00049 \| -0.00014 ± 0.00028 | -0.00010 ± 0.00029 | +0.00005 ± 0.00016 | -0.00030 / -0.00023 | +0.00002 ± 0.00024 \| -0.00027 ± 0.00013 | yes |
| w5_e10_kx05 | 4 | 0.39916 \| 0.32237 | 0.9736 | 0.9720 | 0.9759 | 4 | -0.00032 ± 0.00045 \| -0.00061 ± 0.00025 | -0.00011 ± 0.00025 | -0.00021 ± 0.00014 | -0.00006 / +0.00002 | +0.00019 ± 0.00022 \| -0.00027 ± 0.00012 | yes |
| w5_e10_kwwd05 | 8 | 0.39958 \| 0.32266 | 0.9819 | 0.9759 | 0.9788 | 8 | -0.00073 ± 0.00041 \| -0.00068 ± 0.00023 | -0.00012 ± 0.00028 | -0.00049 ± 0.00015 | -0.00011 / -0.00001 | +0.00003 ± 0.00025 \| -0.00027 ± 0.00014 | yes |
| w5_e10_kwh30 | 4 | 0.39899 \| 0.32254 | 0.9754 | 0.9727 | 0.9741 | 4 | -0.00075 ± 0.00047 \| -0.00065 ± 0.00027 | -0.00036 ± 0.00026 | -0.00015 ± 0.00015 | -0.00046 / -0.00052 | +0.00021 ± 0.00021 \| -0.00019 ± 0.00013 | yes |
| w5_e10_kd05 | 4 | 0.39911 \| 0.32229 | 0.9776 | 0.9738 | 0.9749 | 4 | -0.00026 ± 0.00044 \| -0.00058 ± 0.00026 | -0.00015 ± 0.00023 | -0.00020 ± 0.00014 | -0.00021 / -0.00020 | -0.00005 ± 0.00021 \| +0.00001 ± 0.00012 | yes |
| w5_e10_kd05t25 | 4 | 0.39957 \| 0.32221 | 0.9780 | 0.9739 | 0.9748 | 4 | -0.00016 ± 0.00044 \| -0.00090 ± 0.00025 | -0.00016 ± 0.00025 | -0.00032 ± 0.00014 | +0.00005 / +0.00021 | +0.00017 ± 0.00021 \| -0.00008 ± 0.00012 | yes |
| w5_e10_kd08 | 4 | 0.39928 \| 0.32302 | 0.9904 | 0.9786 | 0.9792 | 4 | -0.00106 ± 0.00047 \| -0.00031 ± 0.00026 | -0.00080 ± 0.00025 | -0.00023 ± 0.00014 | -0.00102 / -0.00111 | -0.00003 ± 0.00022 \| +0.00001 ± 0.00011 | no |
| w5_e10_kd03 | 4 | 0.39806 \| 0.32168 | 0.9657 | 0.9680 | 0.9691 | 4 | -0.00134 ± 0.00048 \| -0.00066 ± 0.00028 | -0.00057 ± 0.00027 | -0.00021 ± 0.00016 | -0.00050 / -0.00047 | -0.00004 ± 0.00021 \| +0.00009 ± 0.00013 | no |
| w5_e10_kd10 | 4 | 0.39883 \| 0.32256 | 0.9950 | 0.9784 | 0.9786 | 4 | -0.00205 ± 0.00051 \| -0.00117 ± 0.00029 | -0.00105 ± 0.00029 | -0.00060 ± 0.00016 | -0.00121 / -0.00130 | -0.00026 ± 0.00024 \| -0.00030 ± 0.00013 | no |
| w5_e10_kx08 | 4 | 0.39938 \| 0.32273 | 0.9907 | 0.9771 | 0.9780 | 4 | -0.00102 ± 0.00049 \| -0.00087 ± 0.00028 | -0.00088 ± 0.00028 | -0.00040 ± 0.00016 | -0.00103 / -0.00110 | -0.00021 ± 0.00023 \| -0.00013 ± 0.00013 | no |
| w5_e10_kwaux0 | 8 | 0.39742 \| 0.32107 | 0.9689 | 0.9669 | 0.9680 | 8 | -0.00165 ± 0.00052 \| -0.00133 ± 0.00031 | -0.00080 ± 0.00034 | -0.00045 ± 0.00020 | -0.00093 / -0.00097 | -0.00048 ± 0.00029 \| -0.00023 ± 0.00017 | no |
| w5_e10_kwwd10 | 4 | 0.39894 \| 0.32233 | 0.9848 | 0.9749 | 0.9769 | 4 | -0.00145 ± 0.00049 \| -0.00091 ± 0.00029 | -0.00056 ± 0.00028 | -0.00053 ± 0.00016 | -0.00046 / -0.00045 | +0.00022 ± 0.00023 \| -0.00024 ± 0.00013 | no |
| w5_e10_kw256d2 | 4 | 0.39837 \| 0.32209 | 0.9730 | 0.9717 | 0.9729 | 4 | -0.00140 ± 0.00046 \| -0.00075 ± 0.00026 | -0.00077 ± 0.00026 | -0.00033 ± 0.00015 | -0.00060 / -0.00060 | -0.00036 ± 0.00022 \| -0.00036 ± 0.00012 | no |
| abl_e10_pop1kd05w192 | 4 | 0.39859 \| 0.31643 | 0.9712 | 0.9609 | 0.9622 | 4 | -0.00110 ± 0.00081 \| -0.00636 ± 0.00055 | +0.00056 ± 0.00048 | -0.00183 ± 0.00034 | +0.00116 / +0.00149 | +0.00048 ± 0.00031 \| -0.00046 ± 0.00022 | no |
| abl_e10_pop1kd05t25 | 4 | 0.39821 \| 0.31625 | 0.9759 | 0.9618 | 0.9627 | 4 | -0.00167 ± 0.00082 \| -0.00693 ± 0.00056 | -0.00010 ± 0.00049 | -0.00215 ± 0.00035 | +0.00043 / +0.00081 | +0.00008 ± 0.00032 \| -0.00055 ± 0.00022 | no |
| r6_cd_m02 | 7 | 0.39667 \| 0.31999 | 0.9497 | 0.9592 |  | 4 | -0.00108 ± 0.00042 \| -0.00127 ± 0.00025 | -0.00035 ± 0.00026 | -0.00047 ± 0.00015 | -0.00049 / -0.00058 | -0.00010 ± 0.00030 \| -0.00014 ± 0.00016 | no |
| r6_cd_m02_p70k | 7 | 0.39621 \| 0.32000 | 0.9488 | 0.9577 |  | 4 | -0.00163 ± 0.00045 \| -0.00142 ± 0.00026 | -0.00042 ± 0.00028 | -0.00044 ± 0.00016 | -0.00038 / -0.00048 | +0.00015 ± 0.00031 \| -0.00016 ± 0.00017 | no |
| w5_e10_base | 6 | 0.39648 \| 0.32022 | 0.9484 | 0.9588 | 0.9599 | 4 | -0.00179 ± 0.00043 \| -0.00099 ± 0.00025 | -0.00086 ± 0.00027 | -0.00038 ± 0.00016 | -0.00071 / -0.00062 | -0.00042 ± 0.00028 \| -0.00026 ± 0.00016 | no |
| v23_gru_pre2s | 6 | 0.39481 \| 0.31885 | 0.9413 | 0.9493 |  | 4 | -0.00303 ± 0.00057 \| -0.00200 ± 0.00034 | -0.00082 ± 0.00036 | -0.00053 ± 0.00021 | -0.00081 / -0.00074 | -0.00010 ± 0.00033 \| -0.00021 ± 0.00019 | no |
| v17_pre2s | 7 | 0.39460 \| 0.31911 | 0.9406 | 0.9512 |  | 4 | -0.00305 ± 0.00054 \| -0.00182 ± 0.00031 | -0.00120 ± 0.00034 | -0.00042 ± 0.00020 | -0.00084 / -0.00066 | -0.00049 ± 0.00034 \| -0.00018 ± 0.00019 | no |
| v26_d192_pre2s | 7 | 0.39371 \| 0.31843 | 0.9321 | 0.9467 |  | 4 | -0.00331 ± 0.00054 \| -0.00200 ± 0.00031 | -0.00131 ± 0.00033 | -0.00061 ± 0.00019 | -0.00118 / -0.00119 | -0.00079 ± 0.00034 \| -0.00027 ± 0.00019 | no |
| v22_pre2s_lr1 | 6 | 0.39382 \| 0.31844 | 0.9379 | 0.9448 |  | 4 | -0.00326 ± 0.00062 \| -0.00231 ± 0.00036 | -0.00129 ± 0.00037 | -0.00060 ± 0.00022 | -0.00101 / -0.00106 | -0.00071 ± 0.00035 \| -0.00033 ± 0.00020 | no |
| abl_e10_pop1 | 8 | 0.39467 \| 0.31399 | 0.9488 | 0.9458 | 0.9465 | 8 | -0.00300 ± 0.00087 \| -0.00717 ± 0.00059 | -0.00042 ± 0.00055 | -0.00234 ± 0.00037 | -0.00010 / +0.00021 | +0.00004 ± 0.00045 \| -0.00147 ± 0.00030 | no |
| abl_e10_pop2 | 4 | 0.39472 \| 0.31342 | 0.9481 | 0.9438 | 0.9445 | 4 | -0.00350 ± 0.00089 \| -0.00783 ± 0.00061 | -0.00106 ± 0.00054 | -0.00210 ± 0.00037 | -0.00085 / -0.00060 | +0.00039 ± 0.00036 \| -0.00067 ± 0.00023 | no |
| abl_e10_xdrop_pop | 4 | 0.39403 \| 0.31222 | 0.9521 | 0.9450 | 0.9458 | 4 | -0.00383 ± 0.00090 \| -0.00932 ± 0.00062 | -0.00090 ± 0.00053 | -0.00289 ± 0.00037 | -0.00009 / +0.00037 | -0.00001 ± 0.00035 \| -0.00064 ± 0.00024 | no |

Mean pairwise correlation inside the 16 wide students: 0.9725.

Frontier regression over 56 arms (h = 4): replace = alpha x (arm level - wide level) + beta x (rho_wide - rho_cross). val: alpha 0.462, beta 0.0183, residual sd 0.00020, R2 0.81; holdout: alpha 0.321, beta 0.0051, residual sd 0.00009, R2 0.98; without the popularity-view arms: val alpha 0.480 beta 0.0140, holdout alpha 0.371 beta 0.0079.

Reading.
- Arms within 0.001 of the wide level, popularity views excluded (24 arms): replace on val mean -0.00007, range -0.00051 .. +0.00026, z mean -0.24 (sd 0.73, max +1.11); on the holdout mean -0.00026, max z +1.95; none at z >= 2 on either witness. Two arms are positive on both (kd05w224 +0.00004 | +0.00021; kdmixw192 +0.00026 | +0.00009), both closed by the lead's rule [R: pooled z < 1; kd05w224 +1.38 at matched N]; with 24 arms two such rows are what a null gives.
- Correlation: cross-arm 0.970-0.980 against 0.9725 inside the wide students for 20 of the 24 arms. The four lower values come with a lower member level or a holdout loss (kd03g2w192 0.963 at member 0.39762; hz2 / hz2nF 0.967-0.968 with holdout pool level -0.0029; xd30 0.969 with level -0.0007 | -0.0011). Higher teacher weight or weight decay raises correlation (kd08w192 0.990 within, 0.980 cross) and loses at pool level.
- Same-root re-cools of the same recipe correlate 0.990 with their original run (0.983-0.990 for the other recipes; 0.989 for the full-data hz re-cools on both branches, `out/misc_corr.json`) against 0.972 for a different root: a cooldown re-run is about one third of a new member.
- Popularity-light students are the one arm that is positive at the public / private mix (pop1kd05w192: replace val +0.00056 ± 0.00048, pmix +0.00116 / +0.00149) and negative on the in-period holdout (-0.00183 ± 0.00034); this is the known two-sided picture of the admixture, already used at weight 0.2.
- Hard-label generations (base, F7x, F7x70, F10, v17, v22, v26): cross correlation 0.945-0.959 but member level 0.0027-0.0056 lower; replace -0.0004 .. -0.0013 on val. They stay out.

Frontier regression across arms (same table, h = 4): replace = alpha x (arm pool level - wide pool level) + beta x (rho_wide - rho_cross), fitted with `e8_tables.py` -> `out/frontier.json`: val alpha 0.46, beta 0.018 (R2 0.81, residual sd 0.00020); holdout alpha 0.32, beta 0.005 (R2 0.98). In words: half of a level gap is paid in full by the mixed pool, and one point (0.01) of extra decorrelation at unchanged level buys +0.00018 on val and +0.00005 on the holdout. A second arm that should add +0.0005 on val needs rho_cross near 0.945 at the wide member level. No student arm on disk is below 0.961.

### 2.4 Cooldown branches of one snapshot (two-level law)

Same script, part C. Branches orig / ref / ref2 are three cooldowns of the same 16 plateau snapshots with the same recipe (harness control and its A/A twin [R, E_kd_predeclared.txt blocks 04:16Z and 05:50Z]).

| branches | witness | a | c_root | c_branch | max residual | observed 16x1 | observed 16 x all | 16x1 -> 16x2 | 16x1 -> 16x3 | 16x1 -> 16 x infinity | 16x1 -> 32x1 | 16x1 -> 32x3 | 16x1 -> infinity x 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orig / ref / ref2 (same recipe) | val | 0.40161 | 0.00136 ± 0.00028 | 0.00105 ± 0.00017 | 0.00014 | 0.40152 | 0.40148 | +0.00003 ± 0.00001 | +0.00004 ± 0.00001 | +0.00007 ± 0.00001 | +0.00008 ± 0.00001 | +0.00010 ± 0.00001 | +0.00015 ± 0.00002 |
| orig / ref / ref2 (same recipe) | hold | 0.32448 | 0.00143 ± 0.00016 | 0.00067 ± 0.00009 | 0.00007 | 0.32438 | 0.32444 | +0.00002 ± 0.00000 | +0.00003 ± 0.00000 | +0.00004 ± 0.00001 | +0.00007 ± 0.00001 | +0.00008 ± 0.00001 | +0.00013 ± 0.00001 |
| orig / ref / ref2 / hz / th / hzt | val | 0.40168 | 0.00151 ± 0.00028 | 0.00087 ± 0.00012 | 0.00009 | 0.40157 | 0.40159 | +0.00003 ± 0.00000 | +0.00004 ± 0.00000 | +0.00005 ± 0.00001 | +0.00007 ± 0.00001 | +0.00009 ± 0.00001 | +0.00015 ± 0.00002 |
| orig / ref / ref2 / hz / th / hzt | hold | 0.32433 | 0.00164 ± 0.00016 | 0.00053 ± 0.00007 | 0.00009 | 0.32421 | 0.32413 | +0.00002 ± 0.00000 | +0.00002 ± 0.00000 | +0.00003 ± 0.00000 | +0.00007 ± 0.00001 | +0.00008 ± 0.00001 | +0.00014 ± 0.00001 |

Reading. The branch term is real (c_branch 0.0007-0.0010) but it is divided by R x B: three branches per root add +0.00004 ± 0.00001 on val and +0.00003 on the holdout; infinitely many +0.00007 / +0.00004. The observed 16 x 3 pool is 0.40148 against 0.40152 for 16 x 1 on val and 0.32444 against 0.32438 on the holdout. Doubling the roots is worth twice as much as tripling the branches, and both are below 0.0001.

### 2.5 Draw noise and whether it transfers

`e6_draws.py` -> `out/draws.json`.

| draws | weights | sd of the file level over draws | split-half r | reliability 2r/(1+r) | realised FD16P20 draw excess | expected on new feeds |
|---|---|---|---|---|---|---|
| pop 4 of 12 (base FD16) | val | 0.00013 | -0.156 | -0.371 | +0.00022 | +0.00000 |
| pop 4 of 12 (base FD16) | public | 0.00018 | -0.027 | -0.055 | +0.00044 | +0.00000 |
| pop 4 of 12 (base FD16) | private | 0.00023 | -0.005 | -0.009 | +0.00053 | +0.00000 |
| base 8 of 16 (pop FDP12) | val | 0.00025 | +0.077 | +0.143 |  |  |
| base 8 of 16 (pop FDP12) | public | 0.00032 | +0.111 | +0.200 |  |  |
| base 8 of 16 (pop FDP12) | private | 0.00037 | +0.065 | +0.123 |  |  |
| base 16 of 32 (pop FDP12) | val | 0.00020 | -0.003 | -0.007 |  |  |
| base 16 of 32 (pop FDP12) | public | 0.00024 | +0.022 | +0.043 |  |  |
| base 16 of 32 (pop FDP12) | private | 0.00029 | +0.012 | +0.023 |  |  |

FD16 + all 12 pop minus the mean 4-of-12 file: +0.00007 ± 0.00010.

Reading. The level differences between same-recipe draws on val (sd 0.00013 for the popularity-light draw, 0.00020 for a 16-of-32 base draw, 0.00025 for an 8-of-16 base draw) are not a property of the members: the split-half correlation over users is between -0.16 and +0.11. With 12 or 32 members behind these draws the correlation itself has an SE of roughly 0.1-0.3, so "zero" is the reading, not "negative". Consequence for the picks: the +0.00022 val (+0.00044 public mix, +0.00053 private mix) by which FD16P20 exceeds its expected draw is val-sample noise; the expected level of FD16P20 on new feeds is that of the average draw, and choosing members or member subsets by val has no value (the lead's lesson 2, now with a number).

### 2.6 Which views differ beyond seed noise

`e7_views.py` -> `out/views.json`. Hard-label members against the standard hard-label view (10-epoch arms against `w5_e10_base` x6, base-base correlation 0.9484; 2-epoch ablations against `abl_base`, base-base 0.9778). Excess decorrelation = base-base correlation minus arm-to-base correlation at different seeds.

| arm | member gap val \| hold | corr to base (different seed) | excess decorrelation | matched mix vs base (val \| hold) |
|---|---|---|---|---|
| 2ep:abl_xdrop_pop | -0.00290 \| -0.00737 | 0.9593 | +0.0185 | +0.00078 ± 0.00061 \| -0.00119 ± 0.00039 |
| abl_e10_pop2 | -0.00176 \| -0.00680 | 0.9330 | +0.0154 | -0.00008 ± 0.00053 \| -0.00161 ± 0.00036 |
| 2ep:abl_pop2 | -0.00067 \| -0.00611 | 0.9632 | +0.0145 | +0.00103 ± 0.00062 \| -0.00087 ± 0.00039 |
| abl_e10_xdrop_pop | -0.00245 \| -0.00800 | 0.9342 | +0.0143 | -0.00028 ± 0.00052 \| -0.00233 ± 0.00035 |
| abl_e10_pop1 | -0.00181 \| -0.00623 | 0.9344 | +0.0140 | +0.00001 ± 0.00050 \| -0.00172 ± 0.00033 |
| 2ep:abl_no_retr | -0.00348 \| -0.00293 | 0.9645 | +0.0133 | -0.00047 ± 0.00051 \| -0.00044 ± 0.00029 |
| 2ep:abl_pop1 | -0.00110 \| -0.00599 | 0.9645 | +0.0133 | +0.00096 ± 0.00060 \| -0.00078 ± 0.00038 |
| w5_e10_bs4 | -0.00248 \| -0.00189 | 0.9368 | +0.0116 | -0.00041 ± 0.00054 \| -0.00036 ± 0.00031 |
| v22_pre2s_lr1 | -0.00266 \| -0.00178 | 0.9371 | +0.0113 | -0.00070 ± 0.00033 \| -0.00035 ± 0.00019 |
| v26_d192_pre2s | -0.00277 \| -0.00179 | 0.9376 | +0.0109 | -0.00053 ± 0.00030 \| -0.00041 ± 0.00017 |
| v23_gru_pre2s | -0.00168 \| -0.00137 | 0.9406 | +0.0079 | -0.00045 ± 0.00031 \| -0.00049 ± 0.00019 |
| w5_e10_e20 | -0.00072 \| -0.00051 | 0.9415 | +0.0070 | +0.00009 ± 0.00033 \| -0.00002 ± 0.00019 |
| w5_e10_xf0 | -0.00279 \| -0.00201 | 0.9415 | +0.0069 | -0.00114 ± 0.00040 \| -0.00076 ± 0.00023 |
| w5_e10_pseudo_xpre2 | -0.00070 \| -0.00007 | 0.9423 | +0.0062 | -0.00066 ± 0.00050 \| +0.00028 ± 0.00028 |
| 2ep:abl_no_aux | -0.00211 \| -0.00228 | 0.9724 | +0.0053 | -0.00101 ± 0.00043 \| -0.00046 ± 0.00025 |
| 2ep:abl_xdrop_ctxnd | -0.00127 \| -0.00031 | 0.9730 | +0.0048 | -0.00015 ± 0.00041 \| +0.00046 ± 0.00023 |
| 2ep:abl_xdrop_pre2 | -0.00160 \| -0.00051 | 0.9735 | +0.0043 | -0.00033 ± 0.00043 \| +0.00029 ± 0.00023 |
| abl_e10_xdrop_pre2 | -0.00064 \| -0.00047 | 0.9443 | +0.0041 | -0.00007 ± 0.00048 \| -0.00012 ± 0.00028 |
| 2ep:abl_no_set | -0.00190 \| -0.00142 | 0.9743 | +0.0034 | -0.00090 ± 0.00042 \| -0.00010 ± 0.00024 |
| abl_e10_retr32u | -0.00013 \| -0.00005 | 0.9459 | +0.0025 | +0.00046 ± 0.00049 \| -0.00011 ± 0.00028 |
| abl_e10_blocks3 | +0.00043 \| -0.00020 | 0.9461 | +0.0024 | +0.00037 ± 0.00047 \| -0.00016 ± 0.00026 |
| w5_e10_pseudo | +0.00121 \| +0.00044 | 0.9463 | +0.0021 | -0.00005 ± 0.00047 \| -0.00004 ± 0.00027 |
| abl_e10_retr32 | -0.00005 \| -0.00005 | 0.9465 | +0.0019 | +0.00026 ± 0.00046 \| -0.00018 ± 0.00027 |
| 2ep:abl_lnorm_batch | -0.00144 \| -0.00097 | 0.9762 | +0.0016 | -0.00076 ± 0.00039 \| +0.00011 ± 0.00022 |
| 2ep:abl_pre2noid | -0.00081 \| +0.00002 | 0.9763 | +0.0014 | +0.00002 ± 0.00035 \| +0.00031 ± 0.00019 |
| w5_e10_trunc2 | -0.00030 \| -0.00036 | 0.9471 | +0.0013 | -0.00005 ± 0.00031 \| -0.00026 ± 0.00018 |
| 2ep:abl_retr32u | -0.00003 \| +0.00004 | 0.9765 | +0.0012 | +0.00054 ± 0.00040 \| +0.00037 ± 0.00022 |
| w5_e10_trunc | -0.00042 \| -0.00002 | 0.9476 | +0.0009 | -0.00028 ± 0.00031 \| -0.00006 ± 0.00018 |
| abl_e10_xrel_more | -0.00176 \| -0.00050 | 0.9476 | +0.0008 | -0.00084 ± 0.00046 \| -0.00014 ± 0.00027 |
| abl_e10_latedrop | +0.00019 \| +0.00026 | 0.9477 | +0.0007 | +0.00012 ± 0.00047 \| +0.00011 ± 0.00025 |
| w5_e10_ap1 | +0.00062 \| +0.00042 | 0.9479 | +0.0005 | +0.00020 ± 0.00047 \| +0.00013 ± 0.00027 |
| 2ep:abl_no_sim | -0.00024 \| -0.00035 | 0.9774 | +0.0003 | +0.00045 ± 0.00035 \| +0.00033 ± 0.00019 |
| 2ep:abl_retr_cat | +0.00054 \| -0.00001 | 0.9774 | +0.0003 | +0.00006 ± 0.00032 \| +0.00036 ± 0.00018 |
| r6_cd_m02 | +0.00019 \| -0.00023 | 0.9483 | +0.0001 | +0.00028 ± 0.00027 \| -0.00019 ± 0.00015 |
| 2ep:abl_no_domain | -0.00036 \| -0.00002 | 0.9780 | -0.0002 | +0.00034 ± 0.00039 \| +0.00042 ± 0.00021 |
| 2ep:abl_retr32 | -0.00042 \| -0.00049 | 0.9781 | -0.0004 | +0.00011 ± 0.00037 \| +0.00015 ± 0.00020 |
| 2ep:abl_xdrop_ctxhn_hs | -0.00043 \| -0.00003 | 0.9783 | -0.0006 | -0.00011 ± 0.00039 \| +0.00054 ± 0.00021 |
| 2ep:abl_xrel_more | -0.00042 \| +0.00014 | 0.9786 | -0.0009 | -0.00025 ± 0.00039 \| +0.00034 ± 0.00021 |
| 2ep:abl_blocks3 | +0.00019 \| +0.00041 | 0.9786 | -0.0009 | +0.00043 ± 0.00038 \| +0.00069 ± 0.00020 |
| 2ep:abl_kv64 | +0.00001 \| +0.00004 | 0.9787 | -0.0009 | +0.00034 ± 0.00037 \| +0.00048 ± 0.00019 |
| 2ep:abl_time | -0.00005 \| +0.00001 | 0.9789 | -0.0012 | +0.00017 ± 0.00039 \| +0.00041 ± 0.00021 |

Reading. Only the popularity input family (pop1, pop2, popularity-free) decorrelates by 0.013-0.019 beyond seed noise. The next ones are the no-retrieval model (+0.013 at a member loss of 0.0035 | 0.0029), batch x4 (+0.012 at -0.0025), and the old families with a different learning rate, width or encoder (+0.008..+0.011 at -0.0017..-0.0028); inputs other than popularity are at -0.001..+0.005 (pre-trained scores +0.004, context +0.005); 20 epochs gives +0.007 at -0.0007. By the frontier regression none of these is predicted to pay for its level gap, and no 10-epoch matched mix is positive on both witnesses by more than 1 SE. The mix column of the 2-epoch rows compares against one base pair (seeds 23, 24) and is positive on the holdout for almost every arm, including null arms (time, kv64): it measures that pair, not the arms; only their correlation column is used. This is the same answer as the correlation table of the students: there is no unused view on disk.

## 3. The largest honest ensemble available today [M]

`e3_forms.py` -> `out/forms.json`; `e3c_pairs.py` -> `out/pairs.json`; `e3d_scan_pairs.py` -> `out/scan_pairs.json`; `e3_nested.py` -> `out/nested.json`; `e3b_rank_rep.py` -> `out/rank_rep.json` (rule in `out/e3b_rule.txt`, written before the run).

**Pre-declared structures (member lists fixed before any number was read, nothing fitted):**
- **A** = base W16 + FD16 (32 wide students) | popularity-light pool = every popularity-light student with a fine-tune (4 held-out d128, 4 held-out wide, 12 full-data wide pop1, 4 full-data d128, 4 full-data pop2, 8 full-data popularity-free = 36) | weight 0.2. 44 of 68 members are trained on all data.
- **B** = base FD16 + FDXHZ16 | popularity-light pool = the 28 full-data students | weight 0.2. All 60 members are trained on all data.

**Scans (every form is listed; none was chosen for a file):** c1 / c2 (popularity-light pool arm-equal / view-equal), d1 / d2 / d3 (12 d128 students, 20 hard-label members, or both, at weight 0.1 inside the base pool), f (rank-z mean, softmax-probability mean, trimmed mean 10% and 25% per side, median inside each of the four sub-pools), each on A and on B = 20 forms; e = 10 learned combiners, nested. "step" rows are descriptive sub-structures (11 rows).

| form | kind | final val | FT branch | TO branch | vs FD16P20 | pmix public | pmix private | vs W8P20 | pmix public / private |
|---|---|---|---|---|---|---|---|---|---|
| FD16P20 (rebuilt) | reference | 0.40271 | 0.40190 | 0.40213 | +0.00000 ± 0.00000 | +0.00000 ± 0.00000 | +0.00000 ± 0.00000 | +0.00058 ± 0.00043 | +0.00075 / +0.00069 |
| W8P20 (rebuilt) | reference | 0.40214 | 0.40167 | 0.40212 | -0.00058 ± 0.00043 | -0.00075 ± 0.00053 | -0.00069 ± 0.00064 | +0.00000 ± 0.00000 | +0.00000 / +0.00000 |
| FD16 (no pop) | reference | 0.40187 | 0.40118 | 0.40161 | -0.00085 ± 0.00031 | -0.00110 ± 0.00039 | -0.00132 ± 0.00047 | -0.00027 ± 0.00048 | -0.00035 / -0.00063 |
| W16 (no pop) | reference | 0.40158 | 0.40070 | 0.40144 | -0.00113 ± 0.00043 | -0.00124 ± 0.00053 | -0.00133 ± 0.00065 | -0.00056 ± 0.00038 | -0.00049 / -0.00064 |
| FD16P20vf | reference | 0.40286 | 0.40190 | 0.40197 | +0.00015 ± 0.00020 | -0.00004 ± 0.00024 | -0.00010 ± 0.00029 | +0.00072 ± 0.00044 | +0.00071 / +0.00059 |
| A: 32 wide + 36 pop students | STRUCTURE | 0.40247 | 0.40162 | 0.40183 | -0.00024 ± 0.00028 | -0.00048 ± 0.00034 | -0.00047 ± 0.00041 | +0.00033 ± 0.00036 | +0.00027 / +0.00022 |
| B: all-data 32 + 28 | STRUCTURE | 0.40216 | 0.40144 | 0.40203 | -0.00056 ± 0.00027 | -0.00081 ± 0.00035 | -0.00084 ± 0.00044 | +0.00002 ± 0.00044 | -0.00006 / -0.00015 |
| step: FD16 + FDP12 | step | 0.40256 | 0.40165 | 0.40179 | -0.00016 ± 0.00018 | -0.00052 ± 0.00022 | -0.00070 ± 0.00027 | +0.00042 ± 0.00044 | +0.00023 / -0.00001 |
| step: FD16 + FDP12 + FDPd | step | 0.40258 | 0.40166 | 0.40193 | -0.00013 ± 0.00019 | -0.00040 ± 0.00023 | -0.00055 ± 0.00029 | +0.00044 ± 0.00044 | +0.00034 / +0.00014 |
| step: FD16 + FDP12 + FDPd + FDP2 | step | 0.40255 | 0.40168 | 0.40202 | -0.00016 ± 0.00020 | -0.00043 ± 0.00024 | -0.00054 ± 0.00029 | +0.00041 ± 0.00043 | +0.00032 / +0.00015 |
| step: FD16 + all 28 pop | step | 0.40244 | 0.40183 | 0.40198 | -0.00027 ± 0.00021 | -0.00043 ± 0.00026 | -0.00048 ± 0.00031 | +0.00031 ± 0.00044 | +0.00032 / +0.00021 |
| step: FD16 + FDXHZ, pop FDP4 | step | 0.40238 | 0.40152 | 0.40195 | -0.00034 ± 0.00025 | -0.00045 ± 0.00034 | -0.00041 ± 0.00042 | +0.00024 ± 0.00045 | +0.00030 / +0.00028 |
| step: FDXHZ16 + FDP4 (hz replaces) | step | 0.40224 | 0.40128 | 0.40212 | -0.00047 ± 0.00035 | -0.00072 ± 0.00047 | -0.00072 ± 0.00060 | +0.00011 ± 0.00047 | +0.00003 / -0.00003 |
| step: FD16 + FDXHZ, pop FDP12 | step | 0.40216 | 0.40139 | 0.40206 | -0.00055 ± 0.00026 | -0.00082 ± 0.00035 | -0.00085 ± 0.00044 | +0.00002 ± 0.00044 | -0.00007 / -0.00016 |
| step: W16 + FD16, pop FDP4 | step | 0.40270 | 0.40189 | 0.40202 | -0.00002 ± 0.00024 | -0.00002 ± 0.00029 | +0.00003 ± 0.00036 | +0.00056 ± 0.00038 | +0.00073 / +0.00072 |
| step: W16 + FD16, pop P1w + FDP4 | step | 0.40236 | 0.40153 | 0.40193 | -0.00036 ± 0.00027 | -0.00042 ± 0.00033 | -0.00031 ± 0.00041 | +0.00022 ± 0.00037 | +0.00033 / +0.00038 |
| step: W16 + FD16 + FDXHZ + HZL8, pop A | step | 0.40230 | 0.40125 | 0.40219 | -0.00041 ± 0.00029 | -0.00070 ± 0.00037 | -0.00069 ± 0.00045 | +0.00017 ± 0.00038 | +0.00005 / +0.00000 |
| step: W16, pop P1d + P1w | step | 0.40189 | 0.40122 | 0.40173 | -0.00082 ± 0.00039 | -0.00102 ± 0.00049 | -0.00110 ± 0.00060 | -0.00025 ± 0.00032 | -0.00027 / -0.00041 |
| A.c1 pop arm-equal | scan | 0.40233 | 0.40167 | 0.40193 | -0.00038 ± 0.00027 | -0.00054 ± 0.00034 | -0.00052 ± 0.00041 | +0.00020 ± 0.00036 | +0.00021 / +0.00017 |
| A.c2 pop view-equal | scan | 0.40234 | 0.40166 | 0.40188 | -0.00037 ± 0.00028 | -0.00061 ± 0.00035 | -0.00062 ± 0.00042 | +0.00021 ± 0.00037 | +0.00014 / +0.00007 |
| A.d1 + d128 x12 at 0.1 | scan | 0.40237 | 0.40166 | 0.40187 | -0.00034 ± 0.00028 | -0.00068 ± 0.00035 | -0.00067 ± 0.00043 | +0.00023 ± 0.00036 | +0.00007 / +0.00002 |
| A.d2 + hard x20 at 0.1 | scan | 0.40225 | 0.40169 | 0.40188 | -0.00046 ± 0.00028 | -0.00066 ± 0.00035 | -0.00068 ± 0.00042 | +0.00012 ± 0.00037 | +0.00009 / +0.00001 |
| A.d3 + d128 0.1 + hard 0.1 | scan | 0.40202 | 0.40153 | 0.40195 | -0.00070 ± 0.00029 | -0.00100 ± 0.00035 | -0.00106 ± 0.00042 | -0.00012 ± 0.00037 | -0.00025 / -0.00037 |
| B.c1 pop arm-equal | scan | 0.40227 | 0.40143 | 0.40212 | -0.00044 ± 0.00027 | -0.00071 ± 0.00036 | -0.00070 ± 0.00044 | +0.00013 ± 0.00044 | +0.00004 / -0.00001 |
| B.c2 pop view-equal | scan | 0.40235 | 0.40144 | 0.40209 | -0.00036 ± 0.00028 | -0.00064 ± 0.00036 | -0.00068 ± 0.00045 | +0.00021 ± 0.00044 | +0.00011 / +0.00001 |
| B.d1 + d128 x12 at 0.1 | scan | 0.40220 | 0.40163 | 0.40211 | -0.00051 ± 0.00026 | -0.00067 ± 0.00034 | -0.00063 ± 0.00042 | +0.00007 ± 0.00043 | +0.00008 / +0.00006 |
| B.d2 + hard x20 at 0.1 | scan | 0.40201 | 0.40133 | 0.40212 | -0.00070 ± 0.00027 | -0.00085 ± 0.00034 | -0.00080 ± 0.00042 | -0.00013 ± 0.00044 | -0.00010 / -0.00011 |
| B.d3 + d128 0.1 + hard 0.1 | scan | 0.40190 | 0.40136 | 0.40205 | -0.00081 ± 0.00027 | -0.00092 ± 0.00035 | -0.00083 ± 0.00043 | -0.00023 ± 0.00042 | -0.00017 / -0.00014 |
| B.f rank | scan | 0.40248 | 0.40178 | 0.40179 | -0.00024 ± 0.00030 | -0.00064 ± 0.00040 | -0.00081 ± 0.00050 | +0.00034 ± 0.00046 | +0.00011 / -0.00012 |
| B.f softmax | scan | 0.40207 | 0.40139 | 0.40224 | -0.00064 ± 0.00027 | -0.00093 ± 0.00035 | -0.00098 ± 0.00044 | -0.00006 ± 0.00045 | -0.00018 / -0.00029 |
| B.f trim10 | scan | 0.40213 | 0.40151 | 0.40199 | -0.00059 ± 0.00027 | -0.00080 ± 0.00035 | -0.00080 ± 0.00044 | -0.00001 ± 0.00045 | -0.00005 / -0.00011 |
| B.f trim25 | scan | 0.40211 | 0.40131 | 0.40191 | -0.00060 ± 0.00027 | -0.00093 ± 0.00036 | -0.00092 ± 0.00044 | -0.00002 ± 0.00044 | -0.00018 / -0.00022 |
| B.f median | scan | 0.40218 | 0.40148 | 0.40216 | -0.00054 ± 0.00029 | -0.00080 ± 0.00037 | -0.00077 ± 0.00047 | +0.00004 ± 0.00045 | -0.00005 / -0.00008 |
| A.f rank | scan | 0.40269 | 0.40185 | 0.40174 | -0.00002 ± 0.00031 | -0.00039 ± 0.00040 | -0.00058 ± 0.00050 | +0.00055 ± 0.00039 | +0.00036 / +0.00011 |
| A.f softmax | scan | 0.40244 | 0.40159 | 0.40186 | -0.00027 ± 0.00028 | -0.00040 ± 0.00035 | -0.00041 ± 0.00043 | +0.00031 ± 0.00037 | +0.00035 / +0.00028 |
| A.f trim10 | scan | 0.40238 | 0.40163 | 0.40188 | -0.00034 ± 0.00027 | -0.00046 ± 0.00034 | -0.00039 ± 0.00042 | +0.00024 ± 0.00037 | +0.00029 / +0.00030 |
| A.f trim25 | scan | 0.40242 | 0.40173 | 0.40213 | -0.00030 ± 0.00028 | -0.00047 ± 0.00034 | -0.00042 ± 0.00041 | +0.00028 ± 0.00037 | +0.00028 / +0.00027 |
| A.f median | scan | 0.40240 | 0.40155 | 0.40193 | -0.00032 ± 0.00028 | -0.00058 ± 0.00035 | -0.00056 ± 0.00043 | +0.00026 ± 0.00038 | +0.00017 / +0.00014 |

FD16 + 4 of the 12 wide pop1 students, all 495 draws: mean 0.40249, sd 0.00013, min 0.40214, max 0.40286; the realised FD16P20 is above 472 of 495.

Leave-one-arm-out (final val delta when the arm is removed): **A** e10 kd05w192 -0.00008 ± 0.00019, e10 kd05t25w192 -0.00002 ± 0.00020, fd kd05w192 -0.00028 ± 0.00019, fd kd05t25w192 -0.00043 ± 0.00018, e10 pop1 d128 +0.00004 ± 0.00008, e10 pop1 wide -0.00008 ± 0.00007, fd pop1 wide -0.00022 ± 0.00010, fd pop1 d128 -0.00005 ± 0.00006, fd pop2 wide -0.00015 ± 0.00008, fd popfree wide -0.00003 ± 0.00012 ; **B** fd kd05w192 +0.00013 ± 0.00019, fd kd05t25w192 -0.00006 ± 0.00017, fdxhz kd05w192 +0.00014 ± 0.00018, fdxhz kd05t25w192 +0.00024 ± 0.00019, fd pop1 wide +0.00018 ± 0.00012, fd pop1 d128 -0.00002 ± 0.00007, fd pop2 wide +0.00008 ± 0.00008, fd popfree wide -0.00004 ± 0.00014

| structure | final val | vs B | vs A | vs FD16P20 | vs expected FD16P20-type | pmix public / private (vs expected) |
|---|---|---|---|---|---|---|
| FD16 + FDP12 (28 all-data) | 0.40256 | +0.00040 ± 0.00026 | +0.00009 ± 0.00027 | -0.00016 ± 0.00018 | +0.00007 ± 0.00010 | -0.00008 ± 0.00013 / -0.00017 ± 0.00016 |
| FD16 + all 28 all-data pop (44 all-data) | 0.40244 | +0.00029 ± 0.00024 | -0.00003 ± 0.00025 | -0.00027 ± 0.00021 | -0.00005 ± 0.00013 | +0.00001 ± 0.00017 / +0.00005 ± 0.00021 |
| FD16 + FDP12 + FDPF (36 all-data) | 0.40243 | +0.00028 ± 0.00024 | -0.00004 ± 0.00026 | -0.00028 ± 0.00019 | -0.00006 ± 0.00013 | +0.00001 ± 0.00017 / +0.00006 ± 0.00020 |
| W16 + FD16 + FDP4 (36) | 0.40270 | +0.00054 ± 0.00029 | +0.00023 ± 0.00020 | -0.00002 ± 0.00024 | +0.00021 ± 0.00023 | +0.00042 ± 0.00028 / +0.00057 ± 0.00034 |
| W16 + FD16 + FDP12 (44) | 0.40248 | +0.00032 ± 0.00029 | +0.00001 ± 0.00016 | -0.00024 ± 0.00027 | -0.00001 ± 0.00023 | -0.00007 ± 0.00029 / -0.00006 ± 0.00035 |
| W16 + FD16 + all 36 pop = A | 0.40247 | +0.00031 ± 0.00029 | +0.00000 ± 0.00000 | -0.00024 ± 0.00028 | -0.00002 ± 0.00024 | -0.00004 ± 0.00029 / +0.00006 ± 0.00035 |
| B | 0.40216 | +0.00000 ± 0.00000 | -0.00031 ± 0.00029 | -0.00056 ± 0.00027 | -0.00033 ± 0.00022 | -0.00036 ± 0.00029 / -0.00031 ± 0.00036 |

Scan forms read against their own structure (`out/scan_pairs.json`): A: c1 -0.00014 ± 0.00009, c2 -0.00013 ± 0.00010, d1 -0.00010 ± 0.00012, d2 -0.00022 ± 0.00016, d3 -0.00045 ± 0.00019, rank +0.00022 ± 0.00024, softmax -0.00003 ± 0.00012, trim10 -0.00009 ± 0.00011, trim25 -0.00006 ± 0.00015, median -0.00007 ± 0.00018. B: c1 +0.00011 ± 0.00011, c2 +0.00020 ± 0.00011, d1 +0.00005 ± 0.00013, d2 -0.00015 ± 0.00015, d3 -0.00025 ± 0.00020, rank +0.00032 ± 0.00024, softmax -0.00008 ± 0.00012, trim10 -0.00003 ± 0.00011, trim25 -0.00004 ± 0.00017, median +0.00002 ± 0.00020. Largest positive z: +1.75 (view-equal on B; the same form reads -1.28 on A).

Learned combiners (e):

| form | columns | in-sample val | NESTED val | nested vs B | per-fold nested vs B | nested pmix public / private vs B | nested vs A | nested vs FD16P20 | nested vs W8P20 | in-sample vs B | FT share of weight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| car_arm_FD | 16 | 0.40275 | 0.40218 | +0.00002 ± 0.00033 | +0.0017 -0.0006 +0.0002 -0.0006 -0.0006 | +0.00025 / +0.00051 | -0.00029 ± 0.00038 | -0.00053 ± 0.00038 | +0.00004 ± 0.00047 | +0.00060 | 0.46 |
| nnls_arm_FD | 16 | 0.40197 | 0.40166 | -0.00050 ± 0.00035 | -0.0013 -0.0005 +0.0006 -0.0007 -0.0006 | -0.00076 / -0.00082 | -0.00081 ± 0.00039 | -0.00106 ± 0.00038 | -0.00048 ± 0.00049 | -0.00019 | 0.16 |
| ridge_arm_FD | 16 | 0.40250 | 0.40236 | +0.00020 ± 0.00034 | +0.0004 -0.0004 +0.0004 -0.0008 +0.0014 | +0.00013 / +0.00040 | -0.00011 ± 0.00038 | -0.00036 ± 0.00038 | +0.00022 ± 0.00047 | +0.00035 | 0.46 |
| car_member_FD | 120 | 0.40381 | 0.40211 | -0.00004 ± 0.00034 | -0.0003 +0.0006 -0.0005 -0.0007 +0.0007 | +0.00007 / +0.00027 | -0.00036 ± 0.00037 | -0.00060 ± 0.00037 | -0.00002 ± 0.00048 | +0.00166 | 0.49 |
| nnls_member_FD | 120 | 0.40235 | 0.40219 | +0.00003 ± 0.00040 | -0.0004 +0.0005 +0.0005 -0.0009 +0.0005 | +0.00003 / +0.00002 | -0.00028 ± 0.00042 | -0.00052 ± 0.00041 | +0.00005 ± 0.00051 | +0.00019 | 0.16 |
| car_arm_ALL | 40 | 0.40298 | 0.40204 | -0.00012 ± 0.00037 | -0.0006 -0.0009 +0.0006 -0.0002 +0.0006 | -0.00031 / -0.00033 | -0.00043 ± 0.00038 | -0.00068 ± 0.00040 | -0.00010 ± 0.00045 | +0.00083 | 0.41 |
| nnls_arm_ALL | 40 | 0.40183 | 0.40175 | -0.00041 ± 0.00037 | -0.0015 +0.0002 +0.0006 -0.0005 -0.0009 | -0.00074 / -0.00089 | -0.00072 ± 0.00038 | -0.00096 ± 0.00039 | -0.00039 ± 0.00046 | -0.00032 | 0.18 |
| ridge_arm_ALL | 40 | 0.40235 | 0.40186 | -0.00029 ± 0.00034 | -0.0016 -0.0005 +0.0007 +0.0000 -0.0000 | -0.00071 / -0.00086 | -0.00061 ± 0.00033 | -0.00085 ± 0.00037 | -0.00028 ± 0.00042 | +0.00019 | 0.32 |
| car_member_ALL | 264 | 0.40393 | 0.40191 | -0.00025 ± 0.00035 | +0.0000 -0.0010 +0.0006 -0.0010 +0.0001 | -0.00031 / -0.00028 | -0.00056 ± 0.00032 | -0.00081 ± 0.00037 | -0.00023 ± 0.00041 | +0.00177 | 0.44 |
| nnls_member_ALL | 264 | 0.40217 | 0.40132 | -0.00083 ± 0.00042 | -0.0019 -0.0006 -0.0001 -0.0014 -0.0001 | -0.00101 / -0.00109 | -0.00115 ± 0.00041 | -0.00139 ± 0.00043 | -0.00081 ± 0.00048 | +0.00002 | 0.15 |

Method of (e): two columns per member or per arm pool (z of the fine-tuned and of the train-only prediction). Column sets: FD = the 60 all-data members (6 arms), ALL = the 132 members with a fine-tune (20 arms). Caruana: 64 greedy steps with replacement from the empty pool, target NDCG. NNLS: feed-centred labels, row weight 1 / M. Ridge: lambda chosen on inner folds. Outer folds = the fine-tune's fold rule (GroupKFold(5) by user in flat feed order, user-0 feeds as singletons, `nn/finetune.py --u0_singleton`; fold sizes 2,885 / 2,885 / 2,885 / 2,884 / 2,884); no user is on both sides and nothing is tuned on the outer test fold. A bias that no nesting removes: on val the fine-tuned columns are out-of-fold predictions of models that saw 80% of val, on test they are 15-model averages, so val-fitted weights under-weight the fine-tuned branch (NNLS gives it 0.15-0.18 of the weight).

Replication read of the one scan form that stood out (rank pooling):

| member set | mean-z final val | final delta | FT branch | TO branch | pmix public | pmix private | holdout (TO pool) |
|---|---|---|---|---|---|---|---|
| S1 held-out W16 + P1d + P1w | 0.40189 | +0.00029 ± 0.00025 | -0.00019 ± 0.00024 | +0.00032 ± 0.00028 | +0.00016 ± 0.00031 | +0.00013 ± 0.00039 | -0.00028 ± 0.00018 |
| S2 all-data FD16 + FDP12 | 0.40256 | -0.00002 ± 0.00026 | +0.00028 ± 0.00024 | -0.00010 ± 0.00026 | +0.00018 ± 0.00034 | +0.00026 ± 0.00042 |  |
| S3 all-data FDXHZ + FDPF | 0.40226 | -0.00031 ± 0.00024 | +0.00036 ± 0.00026 | -0.00046 ± 0.00025 | -0.00053 ± 0.00030 | -0.00047 ± 0.00038 |  |
| S4 hard-label F7x + F7x70 + F10 | pop1 x8 | 0.40082 | -0.00025 ± 0.00029 | -0.00092 ± 0.00027 | -0.00013 ± 0.00032 | -0.00047 ± 0.00035 | -0.00057 ± 0.00043 | -0.00026 ± 0.00022 |
| ref FD16P20 (FD16 + FDP4) | 0.40271 | -0.00000 ± 0.00026 | +0.00022 ± 0.00027 | -0.00041 ± 0.00027 | -0.00011 ± 0.00034 | -0.00018 ± 0.00043 |  |
| ref W8P20 (W8 + P1d) | 0.40214 | +0.00015 ± 0.00030 | -0.00027 ± 0.00030 | -0.00027 ± 0.00031 | -0.00013 ± 0.00038 | -0.00040 ± 0.00047 | -0.00006 ± 0.00021 |
| A | 0.40247 | +0.00022 ± 0.00024 | +0.00023 ± 0.00024 | -0.00010 ± 0.00024 | +0.00009 ± 0.00031 | -0.00011 ± 0.00038 |  |
| B | 0.40216 | +0.00032 ± 0.00024 | +0.00034 ± 0.00021 | -0.00024 ± 0.00024 | +0.00016 ± 0.00032 | +0.00003 ± 0.00041 |  |

Pooled over S1..S4: -0.00007, z -0.52 -> NOT REPLICATED.

Reading of section 3.
- Against the realised FD16P20 both structures are lower (A -0.9 SE, B -2.1 SE) and both are below -1 SE at the private mix (A -0.00047 ± 0.00041, B -0.00084 ± 0.00044): by the standing pmix veto neither replaces FD16P20. Against the expected FD16P20-type file A is level (-0.00002 ± 0.00024; pmix -0.00004 / +0.00006) and B is lower (-0.00033 ± 0.00022).
- On the FD16 or W16 + FD16 base, every structure that uses all 12 wide pop1 students sits at 0.40243-0.40258, i.e. at the expected level 0.40249; the two structures whose whole popularity-light pool is the first four of them (FD16P20, W16 + FD16 + FDP4) sit at 0.40270-0.40271. The gap is the draw of those four students, not the size of anything.
- Leave-one-arm-out: A moves by -0.00043 .. +0.00004 (the two full-data wide arms matter most: -0.00028 ± 0.00019 and -0.00043 ± 0.00018; every popularity-light arm is within ±0.00022); B moves by -0.00006 .. +0.00024, and it improves when either hz arm is removed (+0.00014 ± 0.00018, +0.00024 ± 0.00019). No single arm makes or breaks either structure.
- Forms tried: 2 structures, 20 scan forms, 10 nested combiners, 11 descriptive step rows, 1 replication rule. Largest nested reading against B: +0.00020 ± 0.00034 (ridge on all-data arm pools). The bar for an extra file (beats B by more than 2 SE nested) is met by nothing; the nearest fixed structure is "W16 + FD16 + FDP4" at +0.00054 ± 0.00029 (z 1.85), which is the favourable four-student draw again.

## 4. Staged files [M]

`e4_build.py` -> `nn_x/w12/ens_1115/stage/` (`out/e4_build.log`). Format = `src/make_sub.make_sub` (2,904,806 rows, 28,595 feeds, columns feed_id / item_id / score as in `subs/cand_FD16P20kd_r4.parquet`, identical row order). Pools are rebuilt straight from the member files in `preds/` (the build_r4 path) and asserted equal to the cache path. Not gated: the lead gates and moves them.

| file | members | all data | sha256 | final val | FT / TO branch val | vs FD16P20: val | pmix public | pmix private | test corr | paired public SE | vs W8P20: val | pmix public | pmix private | test corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage/cand_ENSAkd_stage.parquet` | 32 base + 36 popularity-light, w 0.2 | 44 of 68 | 0ddea3148bb0fa94250fc938eb2f5b8beb358b09fc17af2ecba209197c9724aa | 0.40247 | 0.40162 / 0.40183 | -0.00024 ± 0.00028 | -0.00048 ± 0.00034 | -0.00047 ± 0.00041 | 0.99951 | 0.00036 | +0.00033 ± 0.00036 | +0.00027 ± 0.00044 | +0.00022 ± 0.00053 | 0.99882 |
| `stage/cand_ENSBkd_stage.parquet` | 32 base + 28 popularity-light, w 0.2 | 60 of 60 | 1840dfb71c5798a0e95e68a84ac2f92d833add4e02cd61feb54ba448b400c221 | 0.40216 | 0.40144 / 0.40203 | -0.00056 ± 0.00027 | -0.00081 ± 0.00035 | -0.00084 ± 0.00044 | 0.99959 | 0.00034 | +0.00002 ± 0.00044 | -0.00006 ± 0.00054 | -0.00015 ± 0.00067 | 0.99759 |

Next to each file: `ens_<TAG>kd_{ft,to}_{val,test}.npy` (flat branch files), `<TAG>_members.txt`, `<TAG>_manifest.json` (member lists, sha256 of the parquet and of the four branch files, the read-out above).
Forecast [I]: level line public = 0.0074 + 0.963 x val gives 0.39498 for A and 0.39468 for B (residual sd 0.0006). Relative to FD16P20 (public 0.394892): its val edge over A is the non-transferable draw of section 2.5, so the expected public difference of A is about 0.0000 ± 0.0004 and P(A above the team best 0.3950094) is about 0.4; B is expected about 0.0003 lower, P(B above 0.3950094) about 0.1. Neither is a level bet. A is not an all-data file (24 of its 68 members exclude the 30,000 holdout feeds), which conflicts with the owner's instruction for finals; its only merit is a member draw twice as large.

## 5. What to train next [I, from section 2]

Costs as given: a 10-epoch student 3.4 GPU-hours, its fine-tune 1 hour (4.4 per usable member), 24 jobs at once. A cooldown re-run costs about 0.25 GPU-hours [R, COVERAGE_MAP row "KD re-cool harness"] plus the fine-tune. Forecasts are file-level val; public = 0.963 x val; intervals are ±2 SE of the law.

| option | new jobs | GPU-hours | forecast file gain (val) | interval | gain per 100 GPU-hours | source |
|---|---|---|---|---|---|---|
| same recipe: 16 more full-data wide students (base 16 -> 32) | 16 + 16 FT | 70 | +0.00006 | +0.00004 .. +0.00008 | +0.00008 | T3: cb (1/16 - 1/32) |
| same recipe: 48 more (base 64) | 48 + 48 FT | 211 | +0.00008 | +0.00006 .. +0.00011 | +0.00004 | T3 |
| same recipe: 112 more (base 128) | 112 + 112 FT | 493 | +0.00010 | +0.00007 .. +0.00013 | +0.00002 | T3 |
| popularity-light: 12 more wide pop1 students (12 -> 24) | 12 + 12 FT | 53 | +0.00001 | 0.00000 .. +0.00001 | +0.00001 | T3: cp (1/12 - 1/24) |
| same recipe, cheap: 32 cooldown re-runs of the 16 full-data snapshots (3 branches per root) | 32 + 32 FT | 40 | <= +0.00004 on the train-only branch; fine-tuned branch not measured for this recipe (hz-type re-cools: -0.00038 ± 0.00024 and -0.00019 ± 0.00021) | +0.00002 .. +0.00006 at best | <= +0.00010 | T5, T11 controls |
| second arm at equal level | none on disk qualifies | | 0 | | | T4, T12 |

- **Same recipe, more seeds:** the best rate is the first 16 students, +0.00006 for 70 GPU-hours, which is 0.2 of the paired public SE of a near-copy pair. It cannot be confirmed on public and would not change a rank except by a tie-break. Its real effect is on variance: a 32-member base halves the member-draw component of a file (val sd 0.00020 -> about 0.00014), which matters only if the final board is decided by 0.0001.
- **Second arm:** the requirement is rho_cross <= about 0.945 at the wide member level for +0.0005, or equivalently one more input view like popularity. No arm, recipe or view on disk meets it (T4, T12). More popularity-light students add nothing (cp 0.00014); the pop2 and popularity-free views are the same view (closed today on replication).
- **Fine-tunes of existing held-out members without one** (T11): 76 wide-level members could be fine-tuned for 76 GPU-hours. The train-only branch says the file would move by -0.0003 .. +0.0003 with SE 0.0002, law value +0.00003 .. +0.00011. The two controls where fine-tunes exist warn that the fine-tuned branch of re-cools does not follow the train-only branch (W16 + 8 hz re-cools: TO +0.00018 ± 0.00023, FT -0.00019 ± 0.00021; FD16 + FDXHZ16: TO -0.00018 ± 0.00025, FT -0.00038 ± 0.00024). These would also be held-out members, not all-data ones. Value today: none.

| arm added to the base pool | fine-tunes needed (GPU-h) | TO-preview (FT branch unchanged) | TO branch val | TO branch holdout | law value if seed-like |
|---|---|---|---|---|---|
| kd08w192 x8 | 8 | -0.00026 ± 0.00017 | +0.00004 ± 0.00023 | -0.00012 ± 0.00012 | +0.00004 |
| kd05w192gru x4 | 4 | +0.00016 ± 0.00016 | +0.00012 ± 0.00022 | +0.00008 ± 0.00011 | +0.00003 |
| kd05w224 x4 | 4 | +0.00011 ± 0.00014 | +0.00017 ± 0.00019 | +0.00021 ± 0.00011 | +0.00003 |
| kdmixw192 x4 | 4 | -0.00001 ± 0.00013 | +0.00021 ± 0.00020 | +0.00002 ± 0.00011 | +0.00003 |
| kwh6 x4 | 4 | +0.00014 ± 0.00012 | -0.00001 ± 0.00020 | +0.00018 ± 0.00011 | +0.00003 |
| kwr32 x4 | 4 | +0.00009 ± 0.00014 | +0.00027 ± 0.00021 | +0.00004 ± 0.00012 | +0.00003 |
| kwd2 x4 | 4 | -0.00022 ± 0.00013 | -0.00001 ± 0.00019 | +0.00008 ± 0.00010 | +0.00003 |
| gru + w224 + kdmix x12 | 12 | +0.00023 ± 0.00017 | +0.00026 ± 0.00024 | +0.00024 ± 0.00013 | +0.00006 |
| RC ref@last x16 | 16 | +0.00005 ± 0.00014 | -0.00004 ± 0.00020 | -0.00005 ± 0.00010 | +0.00007 |
| RC ref2@last x16 | 16 | +0.00033 ± 0.00016 | +0.00007 ± 0.00019 | -0.00001 ± 0.00011 | +0.00007 |
| RC th@last x16 | 16 | +0.00003 ± 0.00015 | -0.00011 ± 0.00022 | -0.00031 ± 0.00013 | +0.00007 |
| RC hzt@last x16 | 16 | +0.00008 ± 0.00020 | +0.00013 ± 0.00028 | -0.00037 ± 0.00016 | +0.00007 |
| RC hznFS@last x16 | 16 | +0.00006 ± 0.00020 | +0.00017 ± 0.00027 | -0.00017 ± 0.00014 | +0.00007 |
| RC xd30@last x16 | 16 | +0.00008 ± 0.00021 | -0.00005 ± 0.00030 | -0.00033 ± 0.00014 | +0.00007 |
| RC hz@last s955-958 x8 | 8 | +0.00007 ± 0.00017 | -0.00000 ± 0.00023 | -0.00014 ± 0.00012 | +0.00004 |
| RC ref + ref2 x32 | 32 | +0.00006 ± 0.00017 | +0.00005 ± 0.00020 | -0.00003 ± 0.00011 | +0.00009 |
| all no-FT wide-level arms x76 | 76 | +0.00013 ± 0.00018 | +0.00021 ± 0.00026 | -0.00020 ± 0.00013 | +0.00011 |

Control HZL8 (fine-tunes exist): both branches +0.00011 ± 0.00024; FT branch -0.00019 ± 0.00021; TO branch +0.00018 ± 0.00023; the TO-preview alone would have read +0.00017 ± 0.00016.

Control FDXHZ (fine-tunes exist): both branches -0.00034 ± 0.00025; FT branch -0.00038 ± 0.00024; TO branch -0.00018 ± 0.00025; the TO-preview alone would have read -0.00016 ± 0.00017.

Caveat on T11 [M, `e5b_addcheck.py` -> `out/addcheck.json`]: its holdout column carries a common offset of +0.0001 .. +0.0003 against the "add" column of T4 (same members on the bare base pool), because every row shares one baseline term (the admixture on W16 alone reads -0.00055 ± 0.00021 on the holdout). Examples: kd05w224 -0.00006 ± 0.00012 on the bare pool against +0.00021 ± 0.00011 inside the branch; the 12-member union of GRU, width 224 and kdmix +0.00000 ± 0.00015 against +0.00024 ± 0.00013. On the bare base pool no addition is positive on the holdout (-0.00020 .. +0.00000). The one row of T11 that is positive on both witnesses (the 12-member union: val +0.00026 ± 0.00024, holdout +0.00024 ± 0.00013) is the best of 17 rows and its holdout half is that offset; it is not evidence for a mixed-recipe pool.

- **Recommendation:** no GPU-hour for pool size or pool composition. If GPUs are idle after the running lanes, 16 more full-data wide students are the only ensemble item with a positive, bounded forecast (+0.00006, variance halved); it should be queued behind anything that changes what a member knows.

## 6. Web sources [W] (generic searches; no project data left the machine)

- Lobacheva, Chirkova, Kodryan, Vetrov, "On Power Laws in Deep Ensembles", NeurIPS 2020, https://arxiv.org/abs/2007.08483 : ensemble quality follows a power law in ensemble size and the fitted law forecasts the gain of larger ensembles from a few trained networks. The 1 / n law of T2 is the same device; here the forecast is +0.00015 in total.
- Fort, Hu, Lakshminarayanan, "Deep Ensembles: A Loss Landscape Perspective", 2019, https://arxiv.org/abs/1912.02757 : independent initialisations reach different modes, while functions along one optimisation trajectory or its subspace cluster in one mode. Matches T4 / T5: a cooldown re-run of the same snapshot correlates 0.990 with its root against 0.972 for a new root.
- Allen-Zhu, Li, "Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning", ICLR 2023, https://arxiv.org/abs/2012.09816 (text read at https://ar5iv.labs.arxiv.org/html/2012.09816): an ensemble over models trained by knowledge distillation "does not exceed the ensemble over the original single models"; distilled models "have learned most of the features from the ensemble" and have less variety. Matches T1: students correlate 0.972 against 0.948 for hard-label members, and a student pool gains 0.0019-0.0021 from 1 to 8 members where a hard-label pool gains 0.0039 from 1 to 7.
- Abe, Buchanan, Pleiss, Cunningham, "Pathologies of Predictive Diversity in Deep Ensembles", TMLR 2024, https://arxiv.org/abs/2302.00704 : interventions that raise diversity at the cost of member accuracy harm ensembles of large networks; more accurate, less diverse members are preferred. Matches the frontier regression (level coefficient 0.46 against 0.018 per unit of decorrelation).
- Gontijo-Lopes, Dauphin, Cubuk, "No One Representation to Rule Them All", ICLR 2022, https://arxiv.org/abs/2110.12899 : error correlation falls in the order re-initialisation, hyper-parameters, architectures, frameworks, datasets; large ensemble gains need a different training methodology or different data. Matches T12: only a different input family (popularity) decorrelates beyond seed noise.
- Wenzel, Snoek, Tran, Jenatton, "Hyperparameter Ensembles for Robustness and Uncertainty Quantification", NeurIPS 2020, https://arxiv.org/abs/2006.13570 : hyper-parameter diversity plus seeds beats seeds alone in image classification. Not reproduced here: under a shared teacher the hyper-parameter arms correlate like seeds (T4).
- Purucker, Beel, "CMA-ES for Post Hoc Ensembling in AutoML", AutoML 2023, https://arxiv.org/abs/2307.00286 , and Purucker et al., https://arxiv.org/abs/2307.08364 : post hoc weight optimisation overfits the validation data for ROC AUC (a ranking metric), and population-based selection beats greedy selection "only statistically significant on validation data". Matches T8 (in-sample up to +0.0018, nested 0).
- Abe et al., "Deep Ensembles Work, But Are They Necessary?", NeurIPS 2022, https://nips.cc/virtual/2022/poster/53960 : ensemble benefits track in-distribution accuracy rather than a separate robustness effect. Consistent with the same law constant under the public and private mix (section 2.1).

## 7. Closed rows this study touches (nothing here re-opens them)

- COVERAGE_MAP "More same-recipe student seeds (W16, W24, W32), students of several teachers, narrow + wide mixes, hard-label families added to the student pool: CLOSED 11-14 for LEVEL": confirmed with 32 students, with the fine-tuned branch and at file level (T2, T3).
- "second-level combiners ... 78 arms: CLOSED 11-13" and "GBDT members, stackers, rank / RRF fusion: CLOSED": confirmed on today's members at final-file level, nested (T8, T9).
- "hz ... CLOSED as a file recipe 11-15": the fourth member set (FD16 + FDXHZ16 as one base) has the same shape (T6 step rows, T11 controls).
- "View diversity inside the popularity-light pool: CLOSED 11-15": confirmed (cp 0.00014; c1 / c2 scans).
- "snapshot / SWAG" and "plateau weight averaging": the prediction-level version (cooldown branches) is measured here for the first time and is worth <= +0.00007 (T5).
- New measurement that bears on a pick: the val edge of FD16P20's four popularity-light students has zero split-half reliability (T10). The row "more wide pop1 students (8 / 12 instead of 4): -0.0002" compared the 12-student file with that favourable draw; against the expected draw the 12-student file is +0.00007 ± 0.00010. This does not make the 12-student file a gain; it says the two are equal in expectation.

## 8. Files

- Scripts: `nn_x/w12/ens_1115/ens_common.py`, `e0_cache.py`, `e1_inventory.py`, `e1b_outside.py`, `e2_law.py`, `e2_arms.py`, `e2c_file_law.py`, `e3_forms.py`, `e3b_rank_rep.py`, `e3c_pairs.py`, `e3d_scan_pairs.py`, `e3_nested.py`, `e4_build.py`, `e2d_misc.py`, `e5_ft_preview.py`, `e5b_addcheck.py`, `e6_draws.py`, `e7_views.py`, `e8_tables.py`, `e9_report.py`.
- Outputs: `nn_x/w12/ens_1115/out/` (`inventory.json`, `inventory_arms.md`, `inventory_outside.md`, `law.json`, `arms.json`, `arms_late.json`, `file_law.json`, `frontier.json`, `forms.json`, `pairs.json`, `scan_pairs.json`, `nested.json`, `rank_rep.json`, `e3b_rule.txt`, `draws.json`, `ft_preview.json`, `addcheck.json`, `misc_corr.json`, `views.json`, `tables.md`, logs `e*.log`).
- Staged: `nn_x/w12/ens_1115/stage/cand_ENSAkd_stage.parquet`, `cand_ENSBkd_stage.parquet`, `ens_ENS{A,B}kd_{ft,to}_{val,test}.npy`, `ENS{A,B}_members.txt`, `ENS{A,B}_manifest.json`.
- Cache: `nn_x/w12/ens_1115/cache/` (23 GB of z arrays; can be deleted, `e0_cache.py` rebuilds it in 35 s).
