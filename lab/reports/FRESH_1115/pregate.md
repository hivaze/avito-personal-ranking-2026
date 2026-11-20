# Fresh exposure from the val + test feeds themselves: CPU pre-gate
Assembled 2026-11-15 15:50 UTC by `nn_x/w12/fresh/pregate/assemble_report.py` from the outputs in `nn_x/w12/fresh/pregate/out/` (every number below is read from a file a script wrote; [M] measured here, [D] documented elsewhere, [I] inference). Criteria and their time stamps: `nn_x/w12/fresh/pregate/DECLARATION.txt` (block 1 at 14:54Z before any number; amendments 1-3 stamped). Progress and rule notes: `PROGRESS.md`.

## 0. Verdict: **NO**
- **Decision cell (declared: conditional logit, as-of forms only, 5 user-grouped folds x 3 fold seeds): nested gain -0.00044 ± 0.00015 (z -2.85) [M].** The rule needs >= +0.0005, z >= 2 and a gain above every placebo draw; all three fail. It is below 18 of the 20 placebo draws (placebo mean -0.00015, sd 0.00017, range -0.00051 .. +0.00011). Even fitted and scored on the same feeds (in sample, the most optimistic read) the as-of arm gives -0.00004 ± 0.00017.
- Forms that use later feeds (same day, +-1 day, whole period): -0.00032 ± 0.00033 (z -0.96); all forms together -0.00064 ± 0.00034 (z -1.89); later-only feeds -0.00012 ± 0.00033. Not WEAK either; their in-sample ceilings are +0.00023 / +0.00011 / +0.00026, below the +0.0005 bar before any nesting.
- At the public / private mix the as-of arm reads -0.00064 ± 0.00021 / -0.00067 ± 0.00026; in the coldest feeds (no-train-impression share >= 45%) -0.00097 ± 0.00043. The lever does nothing where staleness hurts most.
- Why [M]: the within-feed AUC of the fresh counts for the click is 0.494-0.506 (train exposure xp_imp_c50: 0.5335); the same read on 36,965 density-matched train feeds gives 0.4991 for the as-of count against 0.5190 for train exposure. Val / test are a thin random-like sample (about 7,000 feeds a day); an appearance in another such feed says the listing is in the candidate pool, not that it is liked.
- Coverage is not the limit [M]: as-of count >= 1 for 10.8% of val, 20.1% of public-like and 28.0% of private-like candidates (cold candidates 9.6 / 16.1 / 23.4%).
- Density parity IS attainable [M]: random-like train feeds (p_random > 0.9) drawn at the val / test daily feed numbers with the val / test feed-size histogram reproduce every window's count distribution (PSI <= 0.003 pooled). A ranker could be trained on such an input; the nested read says it has nothing to learn from it.
- Transfer [M]: the leak-free as-of form is the non-stationary one (coverage 7.3% on the first val day -> 30.5% on the last private day; raw-count PSI val -> private 0.24, within-feed z 2.2); the stationary forms (same day, whole period) are the ones that need later feeds.
- One disagreement, reported and not promoted: the secondary LightGBM family reads the as-of arm at +0.00079 ± 0.00045 (z +1.78) above ITS control and above all 10 of its placebo draws (max +0.00034), but that family sits -0.00301 below the raw pool score; a tree model that cannot re-shape the pool score (LGB-R, post hoc) reads the as-of arm at -0.00002 ± 0.00022 (z -0.10) against its control, above 6 of its 10 placebo draws, and -0.00027 ± 0.00024 against the raw pool score: it does not meet the post-hoc reading declared in AMENDMENT 3 either.
- Recommendation: do not spend GPU or a submission slot on this input. Close as `fresh exposure from val + test feeds: NO at the CPU pre-gate`.

## 1. Step 0: what the current inputs already take from val / test (from code)
- **No impression count over val / test feeds is in the models.** The 34 train-stage inputs are `cvnd_pre2s` = 8 + 20 + 6 columns (`nn_x/abl/train.py:91-98` cvraw / cvnd, `:110-111` cvnd_pre2s); the 42 fine-tune extras are `n2c_pre_v3` (`nn_x/w5/finetune.py:164`, list printed in `nn_x/w5/logs/w5_e10_base_s951_n2cpv3_ft.log`). Neither contains a candidate count of val / test feeds.
- **An in-split impression count exists on disk and is banned:** `it_cand_cnt_split{,_rk,_z,_gap}` = the item's candidate count within the row's own split, own feed and same-user feeds included (`src/features.py:248` bincount of `cand_item_id` per split, `:336-337` saved per 10K feeds). It was in the first fine-tune presets (`nn_x/w5/finetune.py:67` ITEM_PRESET) and is removed by `ITEM_CLEAN` (`finetune.py:87-89`: "drop in-split candidate counts (they include the same user's later feeds)"). Earlier measurements [D]: within-feed AUC 0.5035 on val / 0.4584 on train rows (`reports/featrev_inventory.md:130`), "in-split candidate counts -0.0018" (`reports/STRATEGY.md:183`). The version built here differs: val + test union, other users only, with as-of windows.
- **`it_hist_*_luo` (5 of the 34 inputs, 14 of the 42 extras) DO count interactions from val / test users' histories** [M]: `it_hist_all` = occurrences of the item in the history snapshots of train + val + test (`src/features.py:241-246`; the identity `it_hist_all == train + val + test history counts` is asserted in `s0_facts.py`), and LUO subtracts only the SAME non-zero user's LATER snapshots (`src/features_luo.py:45-51`). So a val row's count includes other users' test-period histories (later than the row). Histories carry no event dates (only order), so the days are bounded by the snapshot times: up to days 20718-20720 (val) and 20722-20725 (test). Size: 2.2% of the count mass; 18.9% of val and 17.1% of test candidates have >= 1 val / test history occurrence; among cold candidates 8.7% / 8.7%, and for 7.2% / 7.0% of cold candidates it is the only history signal. This is a fresh INTERACTION signal (other users acted on the item), not an exposure count.
- Other inputs: `seq_cv_*` use train histories only for val / test rows (`feats_v2/seq_/sources.py:6, 37-41`); `ctx_nd_* / ctx_hn_* / ctx_hs_*` are computed inside the feed (`feats_v2/ctx_/build_ctx.py`, `build_ctx3.py`, `build_ctx4.py`); `pre2_*` are scores of the self-supervised trajectory encoder whose corpus is the history snapshots of train + val + test (`pretrain/build_corpus.py:1-8`, earliest val / test snapshot per user), again histories, not feeds; `xp_imp_c50` (popularity-light students) = log1p(min(train impressions, 50)) (`nn_x/abl/pop1/build_xp_imp_c50.py:33-36`, source `src/features_tte.py:3-4`: all of train for val / test rows). The docstring of `feats_v2/ctx_/build_ctx.py:15` mentions "exploration (val+test) appearances within +-48h" but no such column is built or stored.
- Closest earlier test [D]: C14 cross-feed prediction consensus (other users' earlier-DAY val feeds) stopped at coverage 4.5% (`reports/COVERAGE_MAP.md:271`). With a strict as-of at the 128 s time grid and the val + test union the coverage is 2-6 times that (section 2).
- Conclusion of step 0: a fresh EXPOSURE input does not exist in the 34 inputs -> the pre-gate was run.

## 2. Step 1: build and coverage [M]
Universe: 43,018 feeds (val 14,423 on days 20718 + 20720; test 28,595 on days 20722-20725), 4,294,777 candidate rows, no (feed, item) duplicate; 219,526 items shared by val and test; 92 non-zero users are in both (all their feeds are excluded for each other). Count for candidate i of feed f (user u, time t) = number of OTHER feeds showing item i whose user differs from u; uid-0 feeds count as different users (variant NZSRC: uid-0 feeds never counted). Windows: **a** strictly earlier (128 s grid), **a24** earlier within 24 h, **b** same Moscow day, **c** +-24 h, **d** whole val + test period, **e** strictly later (diagnostic). Builder `s1_build.py` (kernel checked against a brute-force count on 300 rows), arrays `out/cnt_main.npy`, `out/cnt_nzsrc.npy`. Forms: raw, log1p, within-feed rank and z, indicator, and cold & shown. Public-like = test feeds before `src/pmix.py` BOUND (day 20722 + day 20723 until 11:32 MSK; 8,562 feeds), private-like = the other 20,033.

Share of candidates with count >= 1 / >= 2 / >= 5 (percent), uid-0 feeds as sources:

| window | subset | val | public-like | private-like |
|---|---|---|---|---|
| a as-of | all | 10.8 / 2.8 / 0.3 | 20.1 / 7.1 / 1.1 | 28.0 / 12.0 / 2.6 |
| a as-of | cold | 9.6 / 2.5 / 0.2 | 16.1 / 5.1 / 0.7 | 23.4 / 9.4 / 1.8 |
| a as-of | warm | 11.4 / 2.9 / 0.3 | 23.2 / 8.5 / 1.4 | 32.1 / 14.4 / 3.4 |
| a24 as-of 24 h | all | 7.2 / 1.4 / 0.1 | 8.4 / 1.9 / 0.2 | 13.2 / 3.5 / 0.4 |
| a24 as-of 24 h | cold | 7.5 / 1.7 / 0.1 | 9.3 / 2.3 / 0.2 | 13.6 / 3.9 / 0.5 |
| a24 as-of 24 h | warm | 7.0 / 1.3 / 0.1 | 7.8 / 1.6 / 0.1 | 12.7 / 3.1 / 0.2 |
| b same day | all | 12.9 / 3.4 / 0.3 | 12.9 / 3.4 / 0.4 | 14.1 / 3.9 / 0.4 |
| b same day | cold | 13.4 / 4.0 / 0.5 | 14.2 / 4.2 / 0.5 | 15.3 / 4.6 / 0.6 |
| b same day | warm | 12.7 / 3.1 / 0.2 | 11.9 / 2.9 / 0.2 | 13.1 / 3.3 / 0.3 |
| c +-1 day | all | 12.9 / 3.4 / 0.3 | 17.9 / 5.9 / 0.8 | 20.3 / 7.1 / 1.0 |
| c +-1 day | cold | 13.4 / 4.0 / 0.5 | 19.1 / 6.7 / 1.1 | 21.2 / 7.8 / 1.3 |
| c +-1 day | warm | 12.7 / 3.1 / 0.2 | 17.0 / 5.3 / 0.6 | 19.5 / 6.5 / 0.8 |
| d whole period | all | 31.3 / 14.4 / 3.5 | 35.0 / 16.9 / 4.4 | 34.5 / 16.5 / 4.1 |
| d whole period | cold | 25.5 / 11.0 / 2.2 | 31.8 / 15.0 / 3.5 | 31.2 / 14.4 / 3.2 |
| d whole period | warm | 34.4 / 16.2 / 4.2 | 37.4 / 18.4 / 5.0 | 37.5 / 18.4 / 5.0 |
| e later only | all | 26.6 / 11.4 / 2.6 | 25.4 / 10.4 / 2.1 | 14.7 / 4.6 / 0.6 |
| e later only | cold | 20.5 / 8.0 / 1.4 | 23.8 / 9.7 / 1.9 | 14.8 / 4.7 / 0.7 |
| e later only | warm | 29.8 / 13.2 / 3.2 | 26.7 / 10.9 / 2.2 | 14.7 / 4.5 / 0.6 |

Cold = no train impression: 34.6% of val, 43.6% of public-like, 47.3% of private-like candidates. Without uid-0 sources the as-of coverage is 10.3 / 19.3 / 27.0% (whole period 30.4 / 34.0 / 33.5%).

Coverage (count >= 1, percent; all / cold / warm) by Moscow day:

| day | a | a24 | b | c | d | e |
|---|---|---|---|---|---|---|
| 20718 val | 7.3 / 7.3 / 7.3 | 7.3 / 7.3 / 7.3 | 13.2 / 13.0 / 13.3 | 13.2 / 13.0 / 13.3 | 30.2 / 23.0 / 33.4 | 27.1 / 19.2 / 30.7 |
| 20720 val | 14.6 / 11.7 / 16.5 | 7.0 / 7.7 / 6.5 | 12.6 / 13.7 / 11.9 | 12.6 / 13.7 / 11.9 | 32.6 / 27.7 / 35.6 | 26.0 / 21.7 / 28.8 |
| 20722 public | 18.9 / 14.7 / 22.2 | 6.8 / 7.6 / 6.3 | 12.4 / 13.6 / 11.5 | 16.6 / 17.7 / 15.7 | 34.8 / 31.3 / 37.4 | 26.0 / 24.2 / 27.4 |
| 20723 public / private | 24.1 / 20.3 / 27.3 | 12.6 / 13.3 / 12.1 | 14.5 / 15.8 / 13.3 | 21.6 / 22.7 / 20.7 | 35.6 / 33.0 / 37.9 | 22.7 / 22.1 / 23.2 |
| 20724 private | 27.4 / 23.0 / 31.4 | 13.4 / 13.9 / 13.0 | 13.7 / 14.9 / 12.7 | 21.6 / 22.5 / 20.7 | 34.7 / 31.7 / 37.4 | 16.7 / 16.9 / 16.5 |
| 20725 private | 30.5 / 25.4 / 35.4 | 13.1 / 13.5 / 12.7 | 14.2 / 15.2 / 13.2 | 18.0 / 18.9 / 17.2 | 33.5 / 29.5 / 37.4 | 7.9 / 8.5 / 7.3 |

- The indicator "cold in train but shown elsewhere" (as-of) marks 9.6% of cold val candidates, 16.1% of cold public-like and 23.4% of cold private-like candidates; with the whole period 25.5 / 31.8 / 31.2%.
- Feed level: 88.2% of val feeds have at least one as-of covered candidate (10.4 per feed), 98.5% of private-like feeds (28.2 per feed).
- Cold candidates are covered about as well as warm ones on the same day (b: 13.4% cold vs 12.7% warm on val): the counts do reach the 46% of test candidates that the train-period popularity inputs cannot see.

## 3. Step 2: descriptive read with val labels (in sample; a sanity read, nothing was chosen from it)
Within-feed AUC for the click (mean over feeds with a positive and a negative in the subset, ties 0.5). Raw, log1p, within-feed rank and z of one count are the same ordering inside a feed and share one AUC; only the indicators differ.

| feature | all candidates | cold candidates | warm candidates |
|---|---|---|---|
| pool score (FD16P20) | 0.7334 ± 0.0019 | 0.7594 ± 0.0033 | 0.7125 ± 0.0024 |
| **yardstick xp_imp_c50** (train exposure) | 0.5335 ± 0.0022 | 0.5000 ± 0.0000 | 0.5117 ± 0.0027 |
| it_hist_all_luo | 0.5561 ± 0.0022 | 0.5573 ± 0.0034 | 0.5364 ± 0.0027 |
| fresh a (as-of): count | 0.4970 ± 0.0012 | 0.4913 ± 0.0018 | 0.4990 ± 0.0014 |
| fresh a: cold & shown | 0.4943 ± 0.0006 | 0.4916 ± 0.0018 | 0.5000 ± 0.0000 |
| fresh a24: count | 0.4956 ± 0.0009 | 0.4898 ± 0.0015 | 0.4991 ± 0.0011 |
| fresh b (same day): count | 0.4936 ± 0.0012 | 0.4831 ± 0.0020 | 0.4997 ± 0.0015 |
| fresh c (+-1 day): count | 0.4937 ± 0.0012 | 0.4830 ± 0.0020 | 0.4997 ± 0.0015 |
| fresh d (whole period): count | 0.5032 ± 0.0018 | 0.4938 ± 0.0029 | 0.5049 ± 0.0023 |
| fresh d: cold & shown | 0.4895 ± 0.0010 | 0.4931 ± 0.0028 | 0.5000 ± 0.0000 |
| fresh e (later only): count | 0.5063 ± 0.0018 | 0.5014 ± 0.0027 | 0.5056 ± 0.0022 |
| fresh a, no uid-0 sources | 0.4979 ± 0.0011 | 0.4926 ± 0.0017 | 0.4994 ± 0.0014 |
| fresh d, no uid-0 sources | 0.5042 ± 0.0018 | 0.4962 ± 0.0029 | 0.5051 ± 0.0022 |

- The fresh counts carry no marginal signal (0.49-0.51); among cold candidates the as-of and same-day counts are slightly ANTI-predictive (0.491, 0.483): a cold listing shown in other feeds has a low pool score (mean pool z -0.52 at count 1, -0.67 at 2-4, against -0.24 at 0) and a lower click rate (1.10% vs 1.40%).
- Relative to what the pool already expects (observed / expected clicks, expected = softmax of the pool score): as-of, cold, count 1: 1.077 (376 clicks), count 2-4: 1.040 (95 clicks); warm: 0.993 / 0.994. No departure beyond 1.5 binomial SE: the pool already prices these rows.
- The only departure is in the forms that see LATER feeds, on cold candidates: whole period count 2-4: O/E 1.199 (543 clicks), later-only count 1 / 2-4: 1.119 / 1.177 (783 / 411 clicks). See section 7.

## 4. Step 3: nested residual read (the decision) [M]
Pool score b = within-feed z of 0.5 z(FT) + 0.5 z(train-only) from `preds/ens_FD16P20kd_{ft,to}_val.npy` exactly as `nn_x/w12/kd/build_r4.py`; its plain val NDCG reproduces 0.40271 (build_r4: 0.40271). Model (declared 14:54Z): per-feed conditional logit (listwise softmax, multi-positive), L2 = 10, L-BFGS-B 300 iterations; control = [b, b x ln M, cold, cold x b, cold x ln M]; arm = control + 8 columns per window (count capped at 50, log1p, within-feed rank, within-feed z, indicator, cold & shown, log1p x cold, log1p x ln M). 5 folds grouped by user (uid-0 feeds singletons) x 3 fold seeds; gain = mean over 14,423 feeds of the per-feed NDCG difference (averaged over the 3 seeds), SE paired per feed.

| arm (windows) | columns | nested gain | z | per fold seed | pmix public | pmix private |
|---|---|---|---|---|---|---|
| **A: as-of only (a, a24) = decision** | 16 | -0.00044 ± 0.00015 | -2.85 | -0.00052 / -0.00039 / -0.00040 | -0.00064 ± 0.00021 | -0.00067 ± 0.00026 |
| B: later-using (b, c, d) | 24 | -0.00032 ± 0.00033 | -0.96 | -0.00054 / -0.00014 / -0.00027 | -0.00027 ± 0.00041 | -0.00025 ± 0.00050 |
| ALL (a, a24, b, c, d) | 40 | -0.00064 ± 0.00034 | -1.89 | -0.00092 / -0.00046 / -0.00054 | -0.00084 ± 0.00044 | -0.00090 ± 0.00054 |
| L: later-only (e), diagnostic | 8 | -0.00012 ± 0.00033 | -0.36 | -0.00036 / -0.00006 / +0.00006 | -0.00025 ± 0.00042 | -0.00036 ± 0.00052 |
| A, no uid-0 sources | 16 | -0.00045 ± 0.00016 | -2.89 | -0.00048 / -0.00046 / -0.00042 | -0.00069 ± 0.00022 | -0.00069 ± 0.00028 |
| B, no uid-0 sources | 24 | -0.00015 ± 0.00033 | -0.46 | -0.00020 / -0.00006 / -0.00019 | -0.00015 ± 0.00043 | -0.00016 ± 0.00052 |
| ALL, no uid-0 sources | 40 | -0.00055 ± 0.00034 | -1.59 | -0.00076 / -0.00065 / -0.00024 | -0.00071 ± 0.00044 | -0.00076 ± 0.00055 |
| L, no uid-0 sources | 8 | -0.00004 ± 0.00033 | -0.11 | -0.00022 / -0.00001 / +0.00012 | -0.00012 ± 0.00043 | -0.00025 ± 0.00052 |

Slices of the same per-feed gains (gain ± SE; number of feeds in brackets):

*by feed share of no-train-impression candidates (the task's cold)*

| arm | cold<15% [586] | cold15-30% [4681] | cold30-45% [6460] | cold>=45% [2696] |
|---|---|---|---|---|
| A | +0.00060 ± 0.00055 | -0.00042 ± 0.00025 | -0.00032 ± 0.00023 | -0.00097 ± 0.00043 |
| B | +0.00317 ± 0.00158 | -0.00083 ± 0.00057 | -0.00035 ± 0.00049 | -0.00011 ± 0.00076 |
| ALL | +0.00345 ± 0.00181 | -0.00088 ± 0.00057 | -0.00074 ± 0.00051 | -0.00088 ± 0.00083 |
| L | +0.00111 ± 0.00113 | -0.00036 ± 0.00060 | -0.00016 ± 0.00048 | +0.00012 ± 0.00083 |

*by the pmix cold share (never seen in train as candidate or in a history)*

| arm | pmixcold<15% [1397] | pmixcold15-30% [6861] | pmixcold30-45% [4965] | pmixcold>=45% [1200] |
|---|---|---|---|---|
| A | +0.00003 ± 0.00045 | -0.00022 ± 0.00020 | -0.00076 ± 0.00026 | -0.00093 ± 0.00084 |
| B | +0.00085 ± 0.00101 | -0.00079 ± 0.00043 | -0.00003 ± 0.00062 | -0.00010 ± 0.00118 |
| ALL | +0.00123 ± 0.00109 | -0.00090 ± 0.00043 | -0.00062 ± 0.00065 | -0.00145 ± 0.00132 |
| L | +0.00003 ± 0.00090 | -0.00054 ± 0.00045 | +0.00061 ± 0.00062 | -0.00097 ± 0.00126 |

*by day*

| arm | day20718 [7543] | day20720 [6880] |
|---|---|---|
| A | -0.00030 ± 0.00015 | -0.00059 ± 0.00028 |
| B | -0.00041 ± 0.00043 | -0.00021 ± 0.00050 |
| ALL | -0.00047 ± 0.00043 | -0.00083 ± 0.00053 |
| L | -0.00006 ± 0.00044 | -0.00018 ± 0.00051 |

*by feed size*

| arm | M<50 [5893] | M>=50 [8530] |
|---|---|---|
| A | -0.00063 ± 0.00029 | -0.00031 ± 0.00017 |
| B | -0.00037 ± 0.00060 | -0.00028 ± 0.00037 |
| ALL | -0.00077 ± 0.00062 | -0.00055 ± 0.00038 |
| L | -0.00044 ± 0.00061 | +0.00010 ± 0.00037 |

*by user*

| arm | uid0 [658] | uid!=0 [13765] |
|---|---|---|
| A | -0.00074 ± 0.00038 | -0.00042 ± 0.00016 |
| B | -0.00119 ± 0.00110 | -0.00027 ± 0.00034 |
| ALL | -0.00161 ± 0.00122 | -0.00059 ± 0.00035 |
| L | -0.00158 ± 0.00127 | -0.00005 ± 0.00034 |

*position inside day 20718 (thirds)*

| arm | d18:third1 [2505] | d18:third2 [2510] | d18:third3 [2527] |
|---|---|---|---|
| A | -0.00019 ± 0.00021 | -0.00018 ± 0.00027 | -0.00053 ± 0.00029 |
| B | -0.00044 ± 0.00074 | -0.00009 ± 0.00075 | -0.00072 ± 0.00075 |
| ALL | -0.00032 ± 0.00078 | -0.00057 ± 0.00072 | -0.00054 ± 0.00076 |
| L | +0.00041 ± 0.00075 | -0.00054 ± 0.00077 | -0.00006 ± 0.00074 |

*position inside day 20720 (thirds)*

| arm | d20:third1 [2289] | d20:third2 [2286] | d20:third3 [2303] |
|---|---|---|---|
| A | -0.00128 ± 0.00049 | -0.00041 ± 0.00044 | -0.00010 ± 0.00051 |
| B | -0.00026 ± 0.00088 | -0.00002 ± 0.00085 | -0.00035 ± 0.00087 |
| ALL | -0.00206 ± 0.00095 | -0.00061 ± 0.00093 | +0.00018 ± 0.00087 |
| L | -0.00075 ± 0.00097 | +0.00045 ± 0.00086 | -0.00025 ± 0.00081 |

- The control itself is -0.00040 ± 0.00022 against the raw pool score: fitting even 4 extra coefficients by likelihood costs NDCG at this sample size. The decision contrast is arm minus control, so this does not enter the verdict, but it sets the scale: columns without any item link cost -0.00015 (16 columns) to -0.00036 (40 columns) on average here (section 5).
- Only one slice is positive beyond 1 SE for the as-of arm (feeds with < 15% cold candidates, 586 feeds, +0.00060 ± 0.00055); it is one of 20 slices, and in the >= 45% cold feeds, where the staleness loss sits, the arm reads -0.00097 ± 0.00043.

Information-only reads (declared in AMENDMENT 3 after the first two fold seeds had been seen; they cannot give PROMISING / WEAK):

| read | a (as-of) | a24 | b | c | d | e (later) |
|---|---|---|---|---|---|---|
| MIN: control + [z_w, cold & shown] (2 columns) | -0.00020 ± 0.00008 | -0.00007 ± 0.00005 | -0.00006 ± 0.00008 | -0.00005 ± 0.00008 | -0.00006 ± 0.00028 | -0.00002 ± 0.00032 |
| MINB: MIN + [shown x b, cold & shown x b] (4 columns) | -0.00013 ± 0.00011 | -0.00004 ± 0.00010 | -0.00005 ± 0.00016 | -0.00006 ± 0.00016 | -0.00005 ± 0.00027 | -0.00018 ± 0.00028 |
| in-sample ceiling of MIN (fit = score, optimistic) | -0.00000 ± 0.00009 | +0.00002 ± 0.00006 | +0.00006 ± 0.00010 | +0.00005 ± 0.00010 | +0.00015 ± 0.00030 | +0.00029 ± 0.00034 |

- In-sample ceiling of the full arms (fit and scored on all val feeds, the most optimistic number there is): A -0.00004 ± 0.00017, B +0.00023 ± 0.00036, ALL +0.00011 ± 0.00038, L +0.00026 ± 0.00035.
- Convergence: with 5,000 instead of 300 L-BFGS iterations (fold seed 0) the gains are A -0.00052 (300: -0.00052), B -0.00054 (-0.00054), ALL -0.00092 (-0.00092), L -0.00036 (-0.00036): the iteration cap does not change the reading.

Secondary family (declared, information only): LightGBM lambdarank on top of the pool score (init = beta x b; 15 leaves, 200 rounds).

| family | control vs raw pool | arm A vs control | arm B | arm ALL | arm A vs raw pool | placebo of arm A (10 draws) |
|---|---|---|---|---|---|---|
| LGB (declared C4; b is a feature) | -0.00301 ± 0.00044 | +0.00079 ± 0.00045 (z +1.78) | +0.00028 ± 0.00052 | -0.00020 ± 0.00054 | -0.00222 | mean +0.00013, sd 0.00022, max +0.00034; real above 10 of 10 |
| LGB-R (post hoc; b NOT a feature, 7 leaves, 100 rounds) | -0.00025 ± 0.00023 | -0.00002 ± 0.00022 (z -0.10) | -0.00045 ± 0.00031 | -0.00042 ± 0.00032 | -0.00027 ± 0.00024 | mean -0.00009, sd 0.00012, max +0.00008; real above 6 of 10 |

- The declared LGB family damages the pool score by -0.00301 (trees re-fit b on 11,500 feeds per fold); inside that family the as-of columns win back +0.00079 and beat its placebo, yet the arm with MORE columns (ALL, which contains A) is -0.00020 and every LGB arm stays below the raw pool score. By the declaration this is a disagreement between families, not PROMISING (z +1.78 < 2 in any case).

## 5. Step 4: placebo [M]
**P1, the decision placebo (20 draws):** inside (Moscow day x cold flag x 3-hour block of the feed time) the vector of raw counts is permuted across candidate rows and the within-feed forms are recomputed, then the identical nested read is run. Every row keeps a count vector that really occurs for a same-day, same-cold-status, same-time candidate, but of a random other item (correlation of placebo and real counts 0.020).

| arm | real gain | placebo mean | placebo sd | placebo min .. max | real above n of 20 | (real - mean) / sd |
|---|---|---|---|---|---|---|
| A | -0.00044 | -0.00015 | 0.00017 | -0.00051 .. +0.00011 | 2 | -1.65 |
| B | -0.00032 | -0.00033 | 0.00020 | -0.00072 .. +0.00003 | 9 | +0.07 |
| ALL | -0.00064 | -0.00036 | 0.00023 | -0.00075 .. +0.00002 | 1 | -1.19 |

P1 gains per draw, arm A (seeds 100-119): -0.00021, -0.00008, -0.00045, -0.00021, -0.00006, -0.00014, -0.00033, +0.00011, -0.00025, -0.00035, -0.00007, +0.00004, -0.00005, -0.00012, -0.00024, -0.00024, +0.00005, -0.00051, -0.00004, +0.00009.

- Columns with no item link cost -0.00015 (A, 16 columns) to -0.00036 (ALL, 40 columns) on average. The real as-of arm is at the LOW end of its placebo distribution (above 2 of 20 draws, -1.65 sd): the real columns are not better than shuffled ones; the real later-using arms sit inside theirs.
- **P2, the task's literal placebo (10 draws; information):** the item-id column of the source rows is permuted within each Moscow day and the counts are recomputed for the true items. As declared before running it (DECLARATION D2), this keeps the day's multiset of item ids, so the item link survives: the placebo counts correlate with the real ones at 0.706 (a), 0.999 (b) and 1.000 (d). It only randomises WHEN inside the day the other feeds occur. P2 gains: A mean -0.00032 (range -0.00054 .. +0.00007; real above 4 of 10), B mean -0.00042 (real -0.00032), ALL mean -0.00073 (real -0.00064). The exact within-day timing of the other feeds carries nothing either.

## 6. Step 5: the density question [M] (label-free; nothing was trained on train rows)
How a train-row version can be built: choose a pseudo-period inside train with the calendar of the real one (two val-like days with a gap day, then four test-like days); on each of its days draw a random subsample of train feeds of the val / test size (6,050-7,600 feeds, 2.4-3.6% of a train day); count, for every candidate of a sampled feed, the other sampled feeds of other users that show the item, with the same windows. Built here for period X = train days 20703, 20705 | 20707-20710 (43,015 feeds = 1.37% of train feeds) and, for the cold / warm split, period L = days 20709, 20710 | 20713-20715 (cold = no impression in any train feed of days <= 20708; 36,965 feeds = 1.17%). Three sampling populations: U = all train feeds (mean M 78), R = random-like feeds, p_random > 0.9 (mean M 51), RM = random-like feeds drawn with the val + test feed-size histogram (mean M 100 = val + test).

Period X (real calendar), all rows pooled: zero share / mean count of the train version against the real rows, and PSI over the bins {0, 1, 2, 3-4, 5-9, 10-19, 20+}:

| window | population | train val-like vs val | PSI | train public-like vs public-like | PSI | train private-like vs private-like | PSI |
|---|---|---|---|---|---|---|---|
| a | U | 0.840 / 0.345 vs 0.892 / 0.160 | 0.042 | 0.748 / 0.703 vs 0.799 / 0.362 | 0.043 | 0.667 / 1.186 vs 0.720 / 0.616 | 0.048 |
| a | R | 0.921 / 0.114 vs 0.892 / 0.160 | 0.010 | 0.865 / 0.218 vs 0.799 / 0.362 | 0.034 | 0.802 / 0.394 vs 0.720 / 0.616 | 0.040 |
| a | RM | 0.881 / 0.184 vs 0.892 / 0.160 | 0.001 | 0.797 / 0.383 vs 0.799 / 0.362 | 0.001 | 0.719 / 0.683 vs 0.720 / 0.616 | 0.002 |
| a24 | U | 0.882 / 0.215 vs 0.928 / 0.095 | 0.039 | 0.866 / 0.260 vs 0.916 / 0.117 | 0.041 | 0.805 / 0.412 vs 0.868 / 0.197 | 0.051 |
| a24 | R | 0.946 / 0.071 vs 0.928 / 0.095 | 0.005 | 0.943 / 0.074 vs 0.916 / 0.117 | 0.012 | 0.906 / 0.135 vs 0.868 / 0.197 | 0.014 |
| a24 | RM | 0.919 / 0.111 vs 0.928 / 0.095 | 0.001 | 0.911 / 0.127 vs 0.916 / 0.117 | 0.000 | 0.861 / 0.228 vs 0.868 / 0.197 | 0.002 |
| b | U | 0.802 / 0.431 vs 0.871 / 0.191 | 0.061 | 0.806 / 0.425 vs 0.871 / 0.194 | 0.054 | 0.787 / 0.464 vs 0.859 / 0.215 | 0.059 |
| b | R | 0.902 / 0.142 vs 0.871 / 0.191 | 0.010 | 0.912 / 0.122 vs 0.871 / 0.194 | 0.019 | 0.897 / 0.152 vs 0.859 / 0.215 | 0.014 |
| b | RM | 0.856 / 0.224 vs 0.871 / 0.191 | 0.002 | 0.868 / 0.206 vs 0.871 / 0.194 | 0.000 | 0.849 / 0.254 vs 0.859 / 0.215 | 0.003 |
| c | U | 0.802 / 0.431 vs 0.871 / 0.191 | 0.061 | 0.748 / 0.662 vs 0.821 / 0.309 | 0.060 | 0.718 / 0.763 vs 0.797 / 0.362 | 0.066 |
| c | R | 0.902 / 0.142 vs 0.871 / 0.191 | 0.010 | 0.875 / 0.192 vs 0.821 / 0.309 | 0.026 | 0.851 / 0.251 vs 0.797 / 0.362 | 0.021 |
| c | RM | 0.856 / 0.224 vs 0.871 / 0.191 | 0.002 | 0.817 / 0.328 vs 0.821 / 0.309 | 0.001 | 0.788 / 0.422 vs 0.797 / 0.362 | 0.003 |
| d | U | 0.639 / 1.421 vs 0.687 / 0.755 | 0.044 | 0.591 / 1.760 vs 0.650 / 0.889 | 0.059 | 0.593 / 1.688 vs 0.655 / 0.861 | 0.059 |
| d | R | 0.774 / 0.482 vs 0.687 / 0.755 | 0.044 | 0.751 / 0.540 vs 0.650 / 0.889 | 0.057 | 0.749 / 0.559 vs 0.655 / 0.861 | 0.048 |
| d | RM | 0.682 / 0.809 vs 0.687 / 0.755 | 0.001 | 0.653 / 0.940 vs 0.650 / 0.889 | 0.001 | 0.653 / 0.963 vs 0.655 / 0.861 | 0.003 |
| e | U | 0.700 / 1.074 vs 0.734 / 0.594 | 0.034 | 0.690 / 1.055 vs 0.746 / 0.527 | 0.053 | 0.795 / 0.499 vs 0.853 / 0.244 | 0.044 |
| e | R | 0.815 / 0.367 vs 0.734 / 0.594 | 0.042 | 0.826 / 0.321 vs 0.746 / 0.527 | 0.042 | 0.896 / 0.164 vs 0.853 / 0.244 | 0.018 |
| e | RM | 0.735 / 0.624 vs 0.734 / 0.594 | 0.001 | 0.750 / 0.556 vs 0.746 / 0.527 | 0.001 | 0.848 / 0.279 vs 0.853 / 0.244 | 0.002 |

Coverage per day (count >= 1, percent), population RM against the real days 20718, 20720, 20722-20725:

| window | day 1 | day 2 | day 3 | day 4 | day 5 | day 6 |
|---|---|---|---|---|---|---|
| a train / real | 8.4 / 7.3 | 15.7 / 14.6 | 18.9 / 18.9 | 24.2 / 24.1 | 27.8 / 27.4 | 30.3 / 30.5 |
| a24 train / real | 8.4 / 7.3 | 7.7 / 7.0 | 6.9 / 6.8 | 13.2 / 12.6 | 13.8 / 13.4 | 14.2 / 13.1 |
| b train / real | 14.9 / 13.2 | 13.8 / 12.6 | 12.4 / 12.4 | 14.9 / 14.5 | 15.2 / 13.7 | 15.0 / 14.2 |
| c train / real | 14.9 / 13.2 | 13.8 / 12.6 | 16.8 / 16.6 | 21.8 / 21.6 | 22.6 / 21.6 | 19.1 / 18.0 |
| d train / real | 30.9 / 30.2 | 32.8 / 32.6 | 34.3 / 34.8 | 35.6 / 35.6 | 35.2 / 34.7 | 33.5 / 33.5 |
| e train / real | 27.4 / 27.1 | 25.5 / 26.0 | 25.5 / 26.0 | 23.0 / 22.7 | 17.8 / 16.7 | 8.5 / 7.9 |

Period L, cold and warm candidates separately, the two position-free reads (same-day window b; as-of window a on the first day):

| population | subset | b: train val-like vs val (zero / mean) | PSI | b vs public-like PSI | b vs private-like PSI | first-day a: train vs val day 20718 | PSI |
|---|---|---|---|---|---|---|---|
| U | cold | 0.833 / 0.316 vs 0.866 / 0.212 | 0.014 | 0.051 | 0.040 | 0.924 / 0.114 vs 0.927 / 0.103 | 0.001 |
| U | warm | 0.781 / 0.485 vs 0.873 / 0.180 | 0.103 | 0.067 | 0.071 | 0.861 / 0.263 vs 0.927 / 0.094 | 0.071 |
| R | cold | 0.916 / 0.124 vs 0.866 / 0.212 | 0.027 | 0.016 | 0.016 | 0.959 / 0.055 vs 0.927 / 0.103 | 0.020 |
| R | warm | 0.893 / 0.154 vs 0.873 / 0.180 | 0.004 | 0.010 | 0.011 | 0.936 / 0.085 vs 0.927 / 0.094 | 0.001 |
| RM | cold | 0.885 / 0.199 vs 0.866 / 0.212 | 0.005 | 0.000 | 0.002 | 0.942 / 0.088 vs 0.927 / 0.103 | 0.005 |
| RM | warm | 0.843 / 0.254 vs 0.873 / 0.180 | 0.011 | 0.004 | 0.004 | 0.905 / 0.141 vs 0.927 / 0.094 | 0.009 |

- **Parity is attainable (criterion F3: PSI < 0.25; project rule < 0.5).** Population RM reproduces every window at every position with PSI <= 0.003 pooled and <= 0.011 for cold and warm rows on the same-day window; the per-day coverage curve of the as-of count is matched within about 1 point. Zero shares agree within 1.5 points; the train means are 5-18% higher (a heavier tail of very frequent listings, e.g. window a private-like 0.683 vs 0.616), which the capped / log / rank forms absorb. U (max PSI 0.066) and R (max 0.057) also stay under 0.25 but are visibly off: all-feed sampling gives denser counts (zero share 0.80 vs 0.87 for the same-day window), random-like feeds without the size match give thinner ones.
- Side finding [I]: the val / test feeds look like a size-biased sample of the random-like regime of train (the item-overlap structure is reproduced only when both the regime and the M histogram are matched).
- Limits of such a train version [I]: (1) the counts are defined for feeds of the sampled population; for production-ordered train feeds (most of train) the same recipe gives a different distribution, so parity holds for the random-like feeds (about 11% of train feeds), or needs the query feed drawn like RM; (2) each train row needs a position in a pseudo-period, and train has 12 full days, so only a few independent periods exist; rolling periods with fresh subsamples per epoch are possible; (3) the cold flag of a pseudo-period row must be recomputed against the days before that period (cold share 0.283 / 0.449 / 0.487 for period L against the real 0.346 / 0.436 / 0.473).
- Extra descriptive read on those train rows (RM, period L; 36,965 feeds with a click; train labels, no model): within-feed AUC of the as-of count 0.4991 ± 0.0009 (cold 0.4956, warm 0.4992), same-day 0.4952, whole period 0.5005, later-only 0.5016; the train-exposure yardstick on the same rows 0.5190 ± 0.0013. Same picture as on val with 2.6 times the feeds.

## 7. Step 6: leak and transfer thinking
**(i) Later feeds of other users.**
- Mechanism: the platform's popularity loop (a listing that is clicked now is shown more afterwards) and survival (a listing still shown days later was not sold or withdrawn) both tie the number of LATER feeds to this feed's label without any same-user path. The descriptive read is consistent with this: observed / expected clicks of cold candidates at count 1 / 2-4 are 1.08 / 1.04 for the as-of count but 1.12 / 1.18 for the later-only count and 1.04 / 1.20 for the whole period [M, in sample].
- How much of the nested gain comes from later feeds [M]: GAIN(B) - GAIN(A) = +0.00012 ± 0.00032, GAIN(ALL) - GAIN(A) = -0.00020 ± 0.00032, later-only arm L = -0.00012 ± 0.00033 (MIN read of the later window: -0.00002 ± 0.00032; of the whole period: -0.00006 ± 0.00028). The later part is not a usable gain either: its in-sample O/E of 1.1-1.2 sits on 7.1-8.8% of the rows, and the in-sample ceilings are B +0.00023 ± 0.00036, L +0.00026 ± 0.00035, MIN d +0.00015 ± 0.00030, MIN e +0.00029 ± 0.00034: under +0.0003 before nesting, and zero or negative nested.
- Availability by position [M]: the later-only count is >= 1 for 27.1% of candidates on the first val day, 26.0% on the first test day, 16.7% on day 20724 and 7.9% on the last day (3.7% in its last third). 75.0% of the private-like rows are on the last two days. A model fitted on val, where every row has 4-5 later days of feeds, would meet an input that fades to nothing over the private period. The nested gain by position inside val shows no trend to lean on (as-of arm by third of day 20718: -0.00019 / -0.00018 / -0.00053; day 20720: -0.00128 / -0.00041 / -0.00010; first vs second val day -0.00030 vs -0.00059).
- Policy note [D]: the project's leak rule forbids the same user's later snapshots; other users' later data is already inside the models through `it_hist_*_luo` (section 1). Forms b-d would therefore not be a new kind of leak, only a new kind of non-stationarity.

**(ii) Val is 2 days, test is 4: does the count distribution transfer?** PSI of the raw count (val -> public-like / val -> private-like) and of the within-feed z (10 val-quantile bins):

| window | raw PSI val -> public | raw PSI val -> private | within-feed z PSI val -> public | z PSI val -> private | zero share val / public / private | reading |
|---|---|---|---|---|---|---|
| a as-of | 0.076 | 0.238 | 1.028 | 2.168 | 0.892 / 0.799 / 0.720 | does not transfer |
| a24 | 0.003 | 0.044 | 0.042 | 1.106 | 0.928 / 0.916 / 0.868 | does not transfer |
| b same day | 0.000 | 0.001 | 0.002 | 0.022 | 0.871 / 0.871 / 0.859 | stationary |
| c +-1 day | 0.024 | 0.048 | 0.247 | 0.548 | 0.871 / 0.821 / 0.797 | does not transfer |
| d whole period | 0.007 | 0.005 | 0.057 | 0.047 | 0.687 / 0.650 / 0.655 | stationary |
| e later only | 0.002 | 0.107 | 0.009 | 0.780 | 0.734 / 0.746 / 0.853 | does not transfer |

- The as-of count grows with the position in the period (mean 0.10 on the first val day, 0.33 on the first test day, 0.71 on the last): val rows see 0-1 earlier days, private rows 3-5. Raw PSI val -> private 0.24 (warm rows 0.32), and the within-feed forms shift even more because the share of non-zero rows inside a feed triples. The +-1 day window is degenerate on val (the two val days are not adjacent, so c = b there) and wider on test.
- The same-day and whole-period windows are stationary (PSI <= 0.007 raw), but they are the ones that read later feeds. A stationary AND leak-free form would be "earlier feeds of the same day"; it equals window a on day 20718, where the as-of arm reads -0.00030 ± 0.00015.

## 8. What would change this reading
- More coverage would not: the as-of coverage on val (10.8%) is below the private one (28.0%) and cannot be raised on val, but on the density-matched train rows of section 6 the as-of count has a within-feed AUC of 0.4993 ± 0.0013 on the test-like days (coverage 23.7%) and 0.4991 ± 0.0024 on the last day (coverage 28.3%; cold candidates 0.4951); inside val the as-of gain is at or below zero in every third of both days.
- A different source of exposure. The val / test feeds are about 3% of a train day's feed volume and random-like; the count measures membership in the candidate pool. The staleness problem needs a fresh signal of how users RESPOND to a listing; the only fresh response data in the files are the val / test histories, which the models already read through `it_hist_*_luo` and the encoder scores.
- The single non-null number is the declared LightGBM secondary (as-of arm +0.00079 ± 0.00045 over its own control, above its 10 placebo draws). It is a relative gain inside a family that is -0.00301 below the pool score, and the larger arm that contains the same columns is negative. With trees that cannot re-shape b (LGB-R) the as-of arm reads -0.00002 ± 0.00022 against its control (above 6 of 10 placebo draws) and -0.00027 ± 0.00024 against the raw pool score. Neither family puts the as-of forms above the raw pool score; it is not a reason to continue.

## 9. Protocol notes, deviations, rule breaches
- **Thread cap exceeded once:** the first full run of `s3_nested.py` (PID 583480) used about 20 busy threads (two thread pools) for 3 min 23 s; I stopped it by its exact PID at 15:03:09Z and restarted the identical declared run with 6 + 6 threads. Its two printed fold seeds (all arms below the control) were seen before AMENDMENT 3 was written; this is disclosed there, and the information-only reads added afterwards cannot change the verdict.
- Processes stopped: only my own, by exact PID (583480; 595633 = the slow numpy-path placebo process, after its third draw, replaced by the GIL-free path of the same objective: relative loss difference 2.7e-10, real arms reproduced to 1e-5). No pkill / killall. At most 2 heavy processes at a time; three short light checks (2-4 threads, under 40 s each: gradient check, s6_transfer.py, a timing test) ran beside them.
- Placebo: the task's literal placebo (item ids of the counted feeds permuted within day) does not destroy the item link (shown by arithmetic in DECLARATION D2 before running, confirmed by the 0.999-1.000 correlation of its counts with the real ones). The decision placebo is the profile swap P1, which implements the stated intent ("destroys the item link, keeps the count distribution"); the literal version is reported as P2.
- P1 draws 100-102 were fitted through numpy, 103-119 through the numba objective (same function, same folds).
- pmix cold share: the pmix weights use "never seen in train as candidate or in a history" (mean 0.290 on val), not the task's "no train impression" (0.348). Both slicings are reported (AMENDMENT 1).
- Period L of the density step has a different calendar from the real one (declared; AMENDMENT 2 adds period X with the real calendar). The cold flag of period X is not comparable (one train day before it), so cold / warm parity rests on period L.
- The 300-iteration L-BFGS cap was reached by the larger arms; the 5,000-iteration check (section 4) shows the same gains.
- Writes: only `nn_x/w12/fresh/pregate/` and this report. No GPU, no network, no installs, no submission file, no label in any feature; val labels only in the declared descriptive step 2 and inside user-grouped folds (plus the in-sample ceiling, labelled as such); train labels only in the descriptive AUC of section 6.

## 10. Files
- `nn_x/w12/fresh/pregate/DECLARATION.txt`, `PROGRESS.md`
- scripts: `lib.py` (counting kernel, forms, PSI), `nested.py` (conditional logit, folds), `s0_facts.py`, `s1_build.py`, `s2_auc.py`, `s3_nested.py`, `s3b_lgb.py`, `s3c_min.py`, `s3d_lgbr.py`, `s4_placebo.py`, `s5_density.py`, `s5b_train_auc.py`, `s6_transfer.py`, `assemble_report.py`
- outputs: `out/cnt_main.npy`, `out/cnt_nzsrc.npy` (val rows first, then test rows; columns a, a24, b, c, d, e), `out/rowmeta.npz`, `out/s*.json`, `out/s4_placebo_P{1,2}.jsonl`, `out/s3_perfeed.npz` (per-feed gains); logs in `logs/`
