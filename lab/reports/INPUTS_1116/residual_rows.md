# INVENTORY B: the feature rows closed only by a residual / proxy gate

Written 2026-11-16 01:35-02:30 UTC. CPU only (12 threads per process, at most two processes), read-only on every project file; every new file is in `reports/INPUTS_1116/residual_rows/` (scripts `s01..s13`, `lens.py`, `condlib.py`, `cand_specs.py`; outputs in `out/`). Nothing was submitted, no GPU, no web, no process signalled.

Tags: [M] measured here (script + output file named), [R] record on disk (path), [I] judgement. "+-" = 1 SE paired per feed. val = 14,423 feeds / 1,389,971 candidate rows; holdout = the 30,000 in-period feeds / 4,630,966 rows; train = 3,151,208 feeds / 246,745,836 rows; test = 28,595 feeds / 2,904,806 rows [M `out/disk_scan.tsv`, `data/flat/*/feeds.parquet`].

Pools used as the base of every read here [M, reproduced to the last digit of the record]:
- val TO = train-only pool of the 16 held-out wide students (`w5_e10_kd05w192`, `w5_e10_kd05t25w192`, seeds 951-958): 0.40144;
- val FILE = FD16P20, 0.5 z(FT) + 0.5 z(TO) from `preds/ens_FD16P20kd_{ft,to}_val.npy`: 0.40271;
- holdout = the same 16 students, `nn_x/w5/preds/holdout/`: 0.32460.

## 0. The answer

1. **What was closed, and how.** 70 table lines (section 2) cover the 66 map rows that are information rows closed by a residual, GBDT, MLP, frozen-trunk or CPU-proxy gate, plus the feature store itself. On record they stand for about 1,400 hand-specified reads and 23,234 generated ones (AUTOFE) [I, count of the reports' own statements, section 3]. All but six lines (the trained hard-label screens of l.143, 150, 152, 189, 191, 276) never saw the feature as a trained input, and none saw it in a distilled student.

2. **Those gates could not have found anything of the size we look for.** Measured again here on the current pool with the project's own nested residual form [M `out/calib_{val,hold}.json`]:
   - a pure-noise column reads +0.00003 +- 0.00013 on val; the best of 300 noise columns reads **+0.00042** (z 2.5 by its own SE). On the holdout: sd 0.00005, best of 300 +0.00019. A typical real column has SE 0.0003, so the best of 300 pure-noise reads prints about **+0.0008 at z 2.7**. The record's best single reads (it_tr_coec +0.00060 +- 0.00033, cos_u GBM +0.00073 +- 0.00058, proj_mu +0.00045 +- 0.00022, item trend +0.00031 +- 0.00038) are all below that. "+0.0006 at z 2" is what noise gives.
   - real input blocks, offered to pools trained without them, come back at 15% to 65% of their value on val and 30% to 39% on the holdout (the controls of section 1.2 whose value is resolved). The record's 10-26% (map l.323) is confirmed.

3. **The likelihood-ratio check used on 11-15 is not specific either.** Inputs the ranker ALREADY reads print a plain likelihood ratio of up to 48.9 on val (median 19.5, n = 101 columns), 39.5 on the file, 55.6 on the holdout [M `out/negctl_existing_inputs.txt`]. A column correlated with them inherits that. So "LR 4.4 / 11.4, p 0.036 / 0.0007" for the neighbour-completed co-visitation (map l.348) is not evidence of new information.

4. **A lens that is both specific and sensitive was built and calibrated here** (section 1.3): the Rao score test for adding a column to [pool + 84 columns of the ranker's own inputs], with user-clustered variance (`cz`, N(0,1) under "nothing new").
   - 300 label-free placebo columns: sd 0.95 (val) / 0.99 (holdout), largest |cz| 2.58 / 3.69.
   - an input the pool does read (xp_imp_c50 for the popularity-light students): +0.59 / -0.04.
   - inputs the pool does NOT read: the 8 popularity columns on the popularity-light wide students +4.1 .. +8.7 on val and +6.5 .. +18.8 on the holdout (block worth +0.00134 +- 0.00090 / +0.00636 +- 0.00061 as trained inputs); the encoder scalars +4.6 .. +7.3 / +3.0 .. +8.5 (worth +0.00061 +- 0.00083 / +0.00059 +- 0.00049).
   - known leaks are seen at once: non-LUO `it_hist_all` +6.9 / +40.5, in-sample `it_tr_*` +3.5 / +8.7 .. +10.8, in-split candidate counts +3.1 / +17.8.

5. **Result on 305 candidate columns with arrays on disk (284 with val AND holdout)** [M `out/support_table.txt`, `out/all_columns.tsv`]:
   - **Most families are now closed by a lens that has power**: leaf / GBDT score (C01), streak and EWMA sketches (C02), latent geometry (C03), windowed co-visitation variants (C04, C05, C11), item trend (C07), EMDE (C08), pending favourite (C13), cross-feed consensus (C14), trajectory-kNN label transfer (P4), s4 graph, region popularity, external-knowledge cues, `trd_*`, `clu_*`, the unused `seq_*`, the hz / lagged views, the AUTOFE recipe columns (33 on disk: sd 1.26 / 1.23, largest 2.69 / 2.78). None has |cz| >= 3 on both witnesses; two columns reach 2.5 on both (`cv_bm25` of C05: +2.6 / +3.0; one of the 33 AUTOFE columns: -2.7 / -2.7).
   - **25 columns pass both witnesses with the same sign at |cz| >= 2.5 (noise expects 0.02); 19 at >= 3.** They are six kinds of information, none of it a trained input today:

   | # | information (map line) | column | cz val TO / val FILE / holdout (z or rank form, larger) | stable over val days and the private mix? | parity of the existing build |
   |---|---|---|---|---|---|
   | G1 | whole-sequence co-visitation walk, RP3beta (l.266; the walks of l.348 are the same family) | `rp3b05`, `p3a` | +6.0 / +6.0 / +10.0 | yes: +3.8 / +4.5 by val day, +3.4 at the private weights | as the adopted `seq_cv_*` (causal by day, same user skipped); val + holdout only |
   | G2 | catalogue density: cosine to the nearest much-shown train items (l.341) | `kn_cos1`, `kn_cos20` | -4.0 / -4.6 / -6.7 | sign yes, weaker at the private weights (-1.3) | identical distribution on train sample, holdout, val, test [R]; no full-train array |
   | G3 | item click statistics from train labels (l.140, 234, 263) | `bt_theta` (exploration feeds), `tte_it_coec`, `xp_ctr_c` | +5.4 / +5.3 / +6.9; +3.5 / +3.5 / +6.1 | exploration-feed form yes (+3.0 / +2.5); all-feed form NO (+3.6 on val day 20718, +1.3 two days later) | label statistic; day-causal on train rows, 1 d fresh there, 7-10 d stale on test |
   | G4 | item life cycle in the train candidate logs (l.277) | `gap_first`, `n_days_active` | -5.6 / -5.4 / -3.3; -4.5 / -4.6 / -5.4 | gap_first yes; n_days_active no (0 at the private weights) | raw level differs train vs val (PSI 7); only a within-feed form is usable |
   | G5 | item views by surface, leak-free (l.204) | `X3 = log((n_s1+1)/(n_s2+1))`, `n_s2` | +4.5 / +4.2 / +3.3; -3.8 / -3.5 / -4.3 | yes (+3.9, -3.2 at the private weights) | strictly earlier days; val + holdout only |
   | G6 | the user's own earlier labelled feeds (l.226, 307, 341) | `tte_ui_imp`, `ue_k05_coec` | -3.1 / -3.2 / -3.1; +2.6 / +2.6 / +3.7 | weak | fresh on train (72 h), stale on test (357 h); coverage 0.2% of rows for tte_ui_imp |

   One witness only (val arrays only): implicit-ALS profile cosines `als_exp5*` +4.0 .. +5.1, the graph walks of l.348 +6.3 .. +7.7, the exploration prior x_T +4.5 .. +4.9, `w25_it_dsf` -5.7 (= gap_first) [M `out/valonly.txt`]. `ixh_mf1` reads -4.3 on val and -0.3 on the holdout (z form): not supported.

6. **In NDCG the residual form says little, for these columns and for real inputs alike** [M `out/s07_shortlist.log`, `out/s07b_bundle.log`]. Three reads per block: (a) loglik gain per feed given pool + the ranker's own inputs, (b) nested NDCG of pool + block, (c) coefficients fitted on one witness and applied unchanged to the other.

   | block | (a) nats per feed, val / holdout | (b) nested NDCG, val / holdout | (c) holdout coefficients on val; val coefficients on the holdout | value as trained inputs, val / holdout |
   |---|---|---|---|---|
   | P1 positive control: 8 popularity inputs on the popularity-light wide students | 0.00584 / 0.01397 | +0.00040 +- 0.00065 / +0.00258 +- 0.00042 | +0.00056 +- 0.00069; +0.00258 +- 0.00043 | +0.00134 +- 0.00090 / +0.00636 +- 0.00061 [M] |
   | P2 positive control: 3 encoder scalars on the members trained without them | 0.00220 / 0.00147 | +0.00037 +- 0.00051 / +0.00035 +- 0.00023 | +0.00041 +- 0.00044; -0.00005 +- 0.00025 | +0.00061 +- 0.00083 / +0.00059 +- 0.00049 [M] |
   | N1 negative control: 3 inputs the pool reads | 0 / 0 | +0.00002 +- 0.00038 / +0.00006 +- 0.00017 | -0.00003 +- 0.00032; +0.00019 +- 0.00021 | 0 |
   | G1: the co-visitation walk (2 columns) | 0.00175 / 0.00221 | -0.00020 +- 0.00046 / +0.00055 +- 0.00025 | +0.00001 +- 0.00046; +0.00029 +- 0.00025 | not measured |
   | R2: ranks 1-3 of the short list (6 columns) | 0.00306 / 0.00331 | +0.00032 +- 0.00053 / +0.00046 +- 0.00029 | +0.00045 +- 0.00050; +0.00009 +- 0.00030 | not measured |
   | R1: ranks 1-5 of the short list (9 columns) | 0.00521 / 0.00432 | -0.00005 +- 0.00060 / +0.00039 +- 0.00030 | +0.00112 +- 0.00055; +0.00022 +- 0.00034 | not measured |

   The NDCG reads of the bundle are of the size of the encoder block's (a block worth about +0.0006) and none of these reads separates a real +0.0006 from zero. In information the 9-column bundle is 0.9 of the popularity block on val and 0.3 of it on the holdout, and 2.4 to 2.9 times the encoder block. Only a trained arm prices it: the statement of map l.323, now with a list of which columns carry information and which do not.

7. **Size, honestly.** The two clean controls turn information into pool NDCG at 0.23 and 0.28 per nat on val and at 0.46 and 0.40 on the holdout [M, the two tables]. At those rates the 9-column bundle would be worth +0.0012 to +0.0014 on val and +0.0017 to +0.0020 on the fresh holdout. Discounts [I]: (i) the columns were picked as the largest of 305 on these same two witnesses (18 df of pure noise would already give LR 18 of the 150 / 259); (ii) two other controls convert at about 0 (ctx_hn + ctx_hs: plain LR 41 / 34, value -0.00015 +- 0.00060 / -0.00044 +- 0.00033) and at 0.6 on val / 0 on the holdout (ctx_nd); (iii) a distilled student takes half of its target from a teacher that has not seen the input; (iv) seven of the nine columns are item-history signals, 2-4 days old on val and 7-10 days old on test, and the walk is non-zero on 19.8% of val rows. My central estimate for a trained bundle: **+0.0005 at pool level on val-like inputs (range 0 to +0.0012), about half of that for the file at the private mix; P(pool gain >= +0.0003 on val and holdout) about 0.4, P(file gain >= +0.0003 on private) about 0.25.** That is several times any lever of the 11-15 synthesis (<= +0.0001); the pool estimate is about a sixth of the gap to #1, the file estimate a twelfth.

8. **What the unread information is** [M `out/overlap.txt`, val]. Three independently built collaborative scores read +5 to +7.7 next to pool + inputs: the RP3beta walk, the 3-step graph walk of 11-15 and the implicit-ALS profile cosines. The first two are the same information (RP3 falls to +0.4 .. +1.0 given the walk); ALS is separate (+4.6 after RP3). Catalogue density, exploration-feed click strength, surface split and first-seen gap are each independent of all the others. The reads do not move when 20 non-linear forms of the collaborative inputs are added to the null model, and the walk is non-zero on 19.8% of val rows against 11.3% for the windowed co-visitation input [M `out/robust.txt`]. So the ranker's collaborative channel (windowed co-visitation counts, popularity counts, six encoder scalars) leaves collaborative information unread. This is the one concrete answer found here to "information the model does not read"; the residual gates closed it on 11-13 and 11-15 as "0 or negative as residuals".

9. **Short list for a trained-input bundle** (section 5, ranked): (1) `rp3b05` + `p3a`; (2) `kn_cos1` + `kn_cos20`; (3) `X3` + `n_s2`, leak-free; (4) `bt_theta` (the exploration prior x_T is the same information); (5) `gap_first` as a within-feed rank + `never_seen`; then (6) a day-causal ALS, the only separate family that still needs a real build. CPU cost for train + holdout + val + test with parity: about 3.5 wall-hours of builds at 12 threads for ranks 1-5, plus a lagged twin for the stale witness and half a day to a day of parity checks. **NOT ADMISSIBLE** (section 5.2): non-LUO `it_hist_*`, in-split candidate counts, in-sample `it_tr_*` / `u_tr_*`, train-row `pre_*`, later-history columns, f5 forward echo, hindsight targets, val-label priors outside nested folds, lifelong history and the user's earlier labelled feeds as built (fresh on train, stale on test), all-feed item CTR as built (same defect, and the decay is measured), ALS as built (transductive, no train rows), embedding-change flags (no train analogue), later-using exposure or co-visitation forms.

10. Not done: no trained arm; no test-split arrays for ranks 1, 3, 5; no full-train arrays for anything on the short list; no stale-val read (needs lagged builds). The conditional lens is linear in the column (z and rank form) and was not run on the 94 DPA constructions, the P3 arrays or the FRESH counts.

## 1. Method and calibration

### 1.1 Steps

| step | script | output |
|---|---|---|
| per-candidate arrays on disk (project + the session scratchpad of the 11-13 gates), by first dimension | `s01_disk_scan.py` | `out/disk_scan.tsv` (13,956 arrays) |
| plain residual lens: conditional logit of the click on [z(pool), column], LR, user-clustered z, nested 5-fold NDCG | `lens.py`, `s04_sweep.py` | `out/sweep_{val_TO16,val_FILE,hold_TO16}.jsonl` (482 / 482 / 459 columns) |
| calibration: 5 positive controls, 300 placebo columns | `s03_calibrate.py` | `out/calib_{val,hold}.json`, `out/s03_calibrate_*.log` |
| metadata of the feature store (parsed from `reports/featrev_inventory.md` Appendix A) | `s05_meta.py` | `out/store_meta.tsv` (323 columns) |
| conditional score test given pool + the ranker's own inputs; placebo; positive and negative controls | `s06_cond.py` (`condlib.py`) | `out/cond_*.jsonl`, `out/cond_*.json`, `out/s06_cond_*.log` |
| groups: conditional LR, nested NDCG, cross-witness transfer; the same for two positive controls and a negative one | `s07_shortlist.py`, `s07b_bundle.py` | `out/shortlist.json`, `out/s07_shortlist.log`, `out/shortlist_R.json`, `out/s07b_bundle.log` |
| merged tables | `s08_tables.py` | `out/all_columns.tsv`, `out/support_table.txt`, `out/noise_summary.txt`, `out/negctl_existing_inputs.txt` |
| slices (val day, feed size, cold share, public / private mix weights; holdout periods) | `s09_slices.py` | `out/slices.txt` |
| AUTOFE recipe columns; val-only columns (W25, ALS, pre, ud); overlap between families; richer null model | `s10_afe.py`, `s11_valonly.py`, `s12_overlap.py`, `s13_robust.py` | `out/afe.txt`, `out/valonly.txt`, `out/overlap.txt`, `out/robust.txt` |

Columns read: 284 of the feature store (`data/feats`, `data/feats_new`: every column present for train and val; holdout rows are sliced out of the train array) and 198 arrays left by the gates (CCB testers 1-3, time-gap, enrichment, codes follow-up and exploit, pretrain2 P2 / P4, CatBoost family B blocks a-d, item prior, embedding usability, the graph walks). Forms: within-feed z over present rows (log1p first for counts with a maximum above 20; NaN to 0; clipped at 5) and within-feed percentile rank.

### 1.2 The old residual form, measured on real inputs [M `out/calib_{val,hold}.json`]

Base = pool of members trained WITHOUT the block; true value = pool of the with-input twins minus that pool; "nested" = conditional logit on [pool, block], coefficients on 4 user-grouped folds, NDCG read on the 5th.

| control (members without the block; twins) | split | true value as trained inputs | nested read of the block | share | best single column, nested | plain LR of the block (df) |
|---|---|---|---|---|---|---|
| 8 popularity / co-visitation inputs, hard labels, 10 epochs, 4 seeds (`abl_e10_xdrop_pop`; `w5_e10_base`) | val | +0.00160 +- 0.00102 | +0.00037 +- 0.00066 | 0.23 | +0.00041 +- 0.00042 | 142.1 (8) |
| | holdout | +0.00852 +- 0.00066 | +0.00304 +- 0.00046 | 0.36 | +0.00141 +- 0.00034 | 816.6 (8) |
| the same 8 inputs under the CURRENT recipe: popularity-light wide students vs full-view wide students, seeds 951-954 | val | +0.00134 +- 0.00090 | +0.00087 +- 0.00062 | 0.65 | +0.00079 +- 0.00055 | 119.0 (8) |
| | holdout | +0.00636 +- 0.00061 | +0.00247 +- 0.00042 | 0.39 | +0.00115 +- 0.00030 | 651.6 (8) |
| 5 encoder scalars, hard labels, 10 epochs, 2 seeds (`abl_e10_xdrop_pre2`) | val | +0.00061 +- 0.00083 | +0.00084 +- 0.00048 | not resolved | +0.00081 +- 0.00047 | 154.0 (5) |
| | holdout | +0.00059 +- 0.00049 | +0.00018 +- 0.00022 | 0.30 | +0.00033 +- 0.00021 | 197.2 (5) |
| 10 ctx_nd inputs, 2 epochs, 2 seeds (`abl_xdrop_ctxnd`; `abl_base`) | val | +0.00133 +- 0.00065 | +0.00020 +- 0.00038 | 0.15 | +0.00031 +- 0.00020 | 60.0 (10) |
| | holdout | -0.00003 +- 0.00037 | +0.00011 +- 0.00019 | - | +0.00013 +- 0.00015 | 105.7 (10) |
| ctx_hn + ctx_hs (10 inputs), 2 epochs, 2 seeds | val | -0.00015 +- 0.00060 | -0.00026 +- 0.00039 | - | +0.00009 +- 0.00024 | 41.4 (10) |
| | holdout | -0.00044 +- 0.00033 | -0.00022 +- 0.00015 | - | -0.00004 +- 0.00011 | 34.0 (10) |

The first holdout row reproduces the record (+0.00852 +- 0.00066, map l.323). Two readings: (i) a block worth +0.0013 to +0.0016 on val reads +0.0002 to +0.0009 with an SE of 0.0004 to 0.0007, so the old form cannot separate it from zero; (ii) the last two controls read a large likelihood ratio for blocks whose pool-level value is zero or negative: likelihood detects information, it does not price it.

Placebo on the current pool (150 N(0,1) columns + 150 within-feed permutations of 15 real columns) [M `out/noise_summary.txt`]:

| split | plain LR (1 df): mean / p95 / max | nested NDCG: mean, sd | quantiles 5 / 95 / 99% | max of 300 | share with z >= 2 |
|---|---|---|---|---|---|
| val | 1.02 / 3.57 / 11.92 | +0.00003, 0.00013 | -0.00016 / +0.00028 / +0.00036 | +0.00042 (z 2.48) | 1.3% |
| holdout | 0.92 / 3.39 / 12.69 | 0.00000, 0.00005 | -0.00008 / +0.00007 / +0.00013 | +0.00019 (z 2.64) | 1.3% |

### 1.3 The conditional score test [M `out/s06_cond_*.log`, `out/cond_*.json`]

Null model: conditional logit on z(pool) plus a control block = the 31 distinct pre-computed inputs of `cvnd_pre2s` and the in-model columns that have a copy in `data/feats` (f1..f5, cos_u, dot_u, history similarities, 10 per-event-type similarities, category counts / share / recency, feed category share), each as a within-feed z, 15 of them also as a rank: 84 columns on val, 83 on the holdout. For a candidate column the efficient score is summed over feeds and divided by its user-clustered standard deviation.

| check | val TO | val FILE | holdout |
|---|---|---|---|
| LR of the control block itself given the pool (the pool's weighting of its own inputs is off) | 259.2 (84 df) | 193.6 | 292.1 (83 df) |
| nested NDCG of re-weighting the ranker's own inputs (no gain to take) | -0.00092 +- 0.00064 | -0.00241 +- 0.00060 | -0.00063 +- 0.00027 |
| 300 placebo columns: sd of cz, largest absolute cz, share >= 2 | 0.95, 2.58, 3.0% | 0.95, 2.58, 3.0% | 0.99, 3.69, 3.0% |
| negative control: xp_imp_c50 on the popularity-light students (they read it) | +0.59 | - | -0.04 |
| positive control, 8 popularity columns on the popularity-light students (control block without them), range over the 8 | +4.11 .. +8.74 | - | +6.45 .. +18.79 |
| positive control, the same on the hard-label members without them | +5.71 .. +9.79 | - | +11.42 .. +23.94 |
| positive control, encoder scalars pre2_fe_z / pre2_new_z / pre2_hmax32 on the members without them | +7.25 / +6.23 / +4.64 | - | +8.45 / +7.65 / +3.02 |

Limits of this lens: linear in the two forms of a column (a richer null model with 20 non-linear forms of the collaborative inputs changes nothing, section 4.5); in-sample (one coefficient per column, so the selection effect is the multiplicity of section 3, not over-fit); the holdout is in-period and fresh, so for item-history columns it is the optimistic witness; a column that pool + inputs reproduce (R2 >= 0.98) is marked "copy" and not read.

## 2. Every row: one line each

Legend. T = full train array (246,745,836 rows), H = holdout (4,630,966), V = val (1,389,971), Te = test (2,904,806); "xm" = the 19,741,607 rows of the CatBoost train sample. Paths under `ccb/` = `nn_x/insights/ccb_1113/scratch/`. "Gate on record" is [R] from the map line and the file it cites. "Lens today" is [M]: cz on val TO / val FILE / holdout, the larger of the z and the rank form; "copy" = reproduced by pool + inputs; "-" = no per-candidate array on disk for that split, or not a column. Kind of information uses the grouping words of the task.

### 2.1 Rows of T09 (map l.78, 242)

| map l. | feature | information it carries | kind | built by; arrays | parity / leak status | gate on record | trained equivalent today | lens today |
|---|---|---|---|---|---|---|---|---|
| 78 | hzcv: `seq_cvlag_{invd20,pop,norm}`, `hzcv_{expo_slog,censored,stale_d}` | co-visitation of train rows from a source lagged by 0 / 4 / 8 days, plus censoring offsets | co-visitation (staleness parity) | `feat_new/hzcv.py`; T V Te in `data/feats_new` (H inside T at lag 0) | built for parity, no leak flag; 43% of train feeds lose the dense source (data start Sep 6), so coverage parity breaks the other way; val / test are byte copies of `seq_cv_*` | train-fit LightGBM gate: val +0.00032 +- 0.00071, holdout -0.00057 +- 0.00041; GBM residual on the nn_final4 mix +0.00023 +- 0.00041 | yes: `seq_cv_*`; the trained form of the idea is the hz / hz2 student views (closed as file recipes 11-15) | copy on val; holdout at most +2.9 (hzcv_expo_slog) |
| 78, 242 | ixh: `ixh_{good,lpos,mlogM,dlogM,mf1,df1,ln,has}` | mean production position, feed size and f1 audience of the item in other users' earlier train candidate lists | position / production prior; item exposure | `feat_new/ixh.py`; T V Te | label-free, earlier days only, lag 0 / 4 / 8 on train rows; NaN on 64% of train rows against 35-47% on val / test | gate val +0.00012 +- 0.00072, holdout +0.00007, stale-val -0.00109; GBM residual +0.00053 +- 0.00047; one AdamW seed of an older build (v5_cvix) +0.0010 val / +0.0002 holdout | partly: its count is `tte_it_imp` (= `xp_imp_c50` of the popularity-light view); the item-level position prior is not an input | ixh_mf1 -4.4 / -4.2 / -0.5; ixh_ln -1.5 / -1.7 / -4.7: one witness each |
| 78 | `it_hist_hz_{all,et1..et4}` | LUO popularity, age-gated and density-matched for train rows | item popularity (staleness parity) | `feat_new/it_hist_hz.py`; T V Te | parity restored (zero-lift ratio 0.71 -> 0.99); val / test byte-identical to `it_hist_*_luo` | gate val +0.00111 +- 0.00075 (4-seed re-run -0.00027 +- 0.00058), holdout -0.0010 .. -0.0015; cold / warm trade | yes (`it_hist_*_luo`); trained form = hz2 view | copy on val; holdout at most 1.5 |
| 78 | ud: `ud_{cos_prev,drift,drift_fz,gap_l,has}` | drift of the user vector since the same user's strictly earlier snapshot | user-item affinity (across snapshots) | `feat_new/ud.py`; V Te only (train values were built in memory) | earlier snapshot of the same user (allowed), user 0 never linked; previous feed 2.9 d old on train (median), much older on test: freshness defect | LIN on the nn_final5 mix -0.00008 +- 0.00020, GBM +0.00027 +- 0.00028; train-fit gate -0.00056 +- 0.00069 / +0.00034 +- 0.00042 | no | ud_cos_prev +2.7 / +2.6, ud_drift -2.6 / -2.5 (val only, NaN on 70% of rows) |
| 78 | `src_expo_{h,slog}`, `src_censored` | how long the listing existed inside the co-visitation source window | recency / exposure | `feat_new/research_expo.py`; T V Te | label-free | nested blend +0.00028 +- 0.00041; -0.0006 over base + seq_cv | partly (f4 = listing age is in-model) | -1.1 / -0.2 / +2.9: no |
| 78 | causal and 4-day-lagged `it_hist` | popularity counted only before the row | item popularity | review scripts of 11-11 (scratchpad `featrev/`); no array kept | causal by construction | train-fit GBDT: causal -0.0018 +- 0.0008, lagged -0.0027 +- 0.0008 | yes, in the non-causal LUO form | - |

### 2.2 Rows of T15 (map l.150, 152, 189, 191, 245)

| map l. | feature | information it carries | kind | built by; arrays | parity / leak status | gate on record | trained equivalent today | lens today |
|---|---|---|---|---|---|---|---|---|
| 150 | xrel_more: within-feed rank and z of seq_cv_pop, it_hist_et1_luo, ctx_nd_knn_h, ctx_nd_ucent | no new information: relative versions of 4 inputs | feed composition | flag of `nn_x/abl/train.py`; existing arrays T V Te | as the inputs | TRAINED, hard labels, 10 epochs, one pair: val -0.0019, holdout -0.0007, pmix -0.0032 | yes (raw columns) | copies |
| 152 | scale-ups: retrieval 32, unique + type counts, 3 blocks | capacity, not information | other | - | - | 2-epoch trained screens: -0.0005 / -0.0001 / -0.0004 / +0.0001 val | - | - |
| 189 | retr_cat: second retrieval inside the candidate's category | affinity to history items of the same category | user-item affinity | architecture flag; no array | - | trained 2-epoch screen: val +0.00046 +- 0.00049, holdout 0.0 | partly (retrieval m = 16, category embedding) | - |
| 191 | removal of ctx_hn + ctx_hs | (existing inputs) | feed composition | arrays T V Te | fine | 2-epoch screen: worth +0.0005 val / 0.0 holdout; re-measured here on 2-seed pools: -0.00015 +- 0.00060 / -0.00044 +- 0.00033 | they are inputs | - |
| 245 | GBDT leaf-SVD / GBDT out-of-fold score as an input | a tree's view of 177 columns on 134K exploration feeds | other (label-trained model output) | = C01 below | needs a cross-fitted model for train rows | see C01 | no | c01_score +0.4 / +0.4 / +1.9: nothing beyond pool + inputs |

### 2.3 Rows of T16 before the CCB batch (map l.140-146, 192-233)

| map l. | feature | information it carries | kind | built by; arrays | parity / leak status | gate on record | trained equivalent today | lens today |
|---|---|---|---|---|---|---|---|---|
| 140 | item exploration prior x = log(p_i / p0): (V) val labels out of fold, (T) train feeds with p_random > 0.5, (U) union | click rate of the item on random-like feeds of other users | label statistics (item) | `nn_x/audit/itemprior/audit_itemprior.py`; V (`x_val_T`, `x_val_oof`), Te (`x_test_T`, `x_test_V`); no T, no H | V: val labels, usable only inside nested folds, no train-row analogue; T: whole train window, leave-user-out, not built causally for train rows | nested residual on G3: V -0.00011 +- 0.00016, T +0.00018 +- 0.00023, U +0.00042 +- 0.00029; test-row coverage T 37-40%, V 12-14% | no: no label statistic among the 34 inputs or the 42 extras | x_T +4.9 / +4.8 (val only) |
| 141 | AGE-PROD | training-only logit margin by listing age on production rows | position / production prior (a training offset, not an input) | `nn_x/audit/ageprod/`; lift tables only | - | CPU GBDT proxy: -0.0046 val / -0.0034 holdout | f4 is in-model | - |
| 142 | scores of an ID-aware pre-trained sequence encoder (pre_final, ENS_*) | next-item scores from trajectories with an item-ID table | co-visitation (encoder) | `nn_x/r3/stage0_gate.py`; V Te (`data/pretrain/feats_archive`: 45 / 36 arrays) | train rows of `pre_*` quarantined (content-table leak) | nested residual on G3 -0.00018 +- 0.00020 .. -0.00026 +- 0.00012; zero-shot 0.359 | yes: `pre2_*` (6 inputs); `pre_*` in the fine-tune | pre_* columns are copies of pool + inputs (R2 1.00) or at most +1.6 |
| 146 | external calendar, payday, weather, event priors as category x time tables | category demand by day, hour, weekday | price / geo / category (by time) | `nn_x/audit/external/e02_nested_priors.py`; tables, Te 3 arrays | label-free tables fitted on train | every prior within +-0.0003; joint -0.00034 +- 0.00021; in-sample oracle bound +0.00062 +- 0.00036 | category embedding; no time input | - |
| 192 | trajectory-type clusters of user_emb / history profile (per-cluster weights, FiLM, MoE) | kind of user | other (user segment, constant inside a feed) | `nn_x/pretrain2/P1/`; cluster ids per feed for H 30,000 / V 14,423 / Te 28,595 | label-free | 0 of 8 arms; best val +0.00022 +- 0.00025 | user token | not a within-feed column |
| 193 | size of the part of user_emb not explained by the history, times cos_u | user vector beyond the history | user-item affinity | `nn_x/pretrain2/P2/uresid.py`; V H | cross-fitted ridge, label-free | val -0.00034 +- 0.00033, holdout -0.00006 | cos_u, dot_u in-model | -2.6 / -2.6 / -1.5: no |
| 194 | within-category whitened / CSLS similarity for retrieval | similarity in a re-scaled metric | content | `nn_x/pretrain2/P3/whiten_gate.py`; V H (n x 3 and n x 4 x 5) | label-free | cell AUC +0.02 (needed +0.05), residual -0.0003 | retrieval keys, similarity inputs | not read; the whitened max-cos of l.198 are copies |
| 195 | cross-user trajectory-kNN label transfer (`s_pos`, `s_neg`, `contrast` for item and cluster) | what similar users clicked or ignored | label statistics (cross-user) | `nn_x/pretrain2/P4/uknn_gate.py`; V H, 6 columns | as-of bank; a train build was priced at 10x the first estimate | val -0.0001 .. -0.0003; ignored items click above base rate | no | largest 1.8 / 1.7: no |
| 196 | decoded codes: vertical share, last-event vertical, vertical recency, delivery class | groupings of categories and event types | price / geo / category | `nn_x/codes/synth/logs/x01.log`, `x02.log`; logs only | label-free | 0 of 6 residuals pass; best -0.00003 +- 0.00020 | category, type, source embeddings | - |
| 198 | embedding processing: `chg_d` (item vector changed train -> val), `proj_mu`, ZCA / centred / ABTT max-cos, `nd_act`, cos_u interactions | geometry of the content vectors; vector drift between tables | content | `nn_x/emb/usability/`; V 19 arrays, Te 2; holdout in memory | `chg_*` needs two item tables: no train-row analogue | proj_mu val +0.00045 +- 0.00022, holdout -0.00004; cos_u GBM +0.00073 +- 0.00058 (1.26 SE); ZCA -0.00002 | similarity inputs in-model | chg_d +3.2 / +3.1 (val only); proj_mu -2.3 / -2.0; max-cos forms: copies |
| 201 | region-normalised popularity `rp_lift`, `rp_cnt` (home-region key from f2) | local popularity | item popularity x geo | `nn_x/codes/followup/f01_build.py`; V H | earlier days for holdout rows | nested exactly 0; popularity-matched win 0.50 | popularity inputs | at most 1.3: no |
| 204 | 34 decoding-enabled cues: feed-pull share X1, feed-vs-search contrast X2 / X3, novelty X5, feed appeal Y1, funnel depth, sessions, delivery class x region ... | item views split by surface; user feed propensity; funnel depth | item popularity by surface; user-item affinity | `nn_x/codes/exploit/01..05`, `lensA_funnel`, `lensB_screen`; V 34 arrays, H 13 | "V-exact" variants read other users' LATER snapshots (forward leak, measured by that study); "V-lf" = strictly earlier days | all fail the nested G3 gate (+0.0003 at 2 SE); X1 +0.00033 +- 0.00042 falls to -0.00001 +- 0.00030 leak-free; X3 -0.00008 +- 0.00029 | `it_hist_et1_luo` counts card views without the surface split | **X3 leak-free +4.5 / +4.2 / +3.3; n_s2 -3.8 / -3.5 / -4.3: two witnesses**; X5, Y1, Y3 at most 2.0 |
| 218 | 94 label-free constructions (affinity shapes, feed / user aggregates, relative versions, interactions) + GBDT / MLP residuals | re-expressions and aggregates of the model's inputs | feed composition | `nn_x/audit/dpa_1113/nonlinear/`; V 130 arrays, H 97 | label-free | GBDT[G3 + 92 inputs] - GBDT[G3]: val +0.00005 +- 0.00021, holdout -0.00026; 6 of 94 at z > 2 (2.1 expected); 82 of 94 negative on the holdout | derived from inputs | not read here (the 33 AUTOFE columns, same kind, read below: nothing) |
| 225 | within-category production order as a cleaner K9 teacher | production order inside a category | position / production prior (auxiliary target) | probe scripts (scratchpad `ext/russian/probe/`) | - | probe: same-category excess equal in random-order feeds (+0.0145 vs +0.0130) | K9 auxiliary loss | - |
| 226 | similarity to the user's earlier skipped impressions | what the user saw and ignored | user-item affinity from the user's earlier labelled feeds | scratchpad `ext/features/probe_negfb.py`; logs only | earlier feeds of the same user (allowed); fresh on train, stale on test | skipped items wAUC 0.58 (= interest); nested -0.0006 .. +0.0001 | no | related block a of l.341: ue_k05_coec +2.6 / +2.6 / +3.7 |
| 228 | IDF damping of history items in the similarity aggregation | down-weighting of popular history items | content | `ext/features/probe_histidf.py`; logs only | label-free | +0.00024 +- 0.00025 | similarity inputs | - |
| 229 | typed / backward co-visitation, type-1 share, reference co-visitation | co-visitation by event type and direction | co-visitation | `ext/xcheck169/probe_typedcv.py`; logs only | causal source | all <= +0.00024, mostly negative | `seq_cv_*` | the built variants (C04, C05, C11) read nothing |
| 230 | category-sequence family (runs, momentum, decayed shares) | category momentum in the history | user-item affinity (category) | `ext/xcheck169/probe_catC.py`; logs only | own history | per column <= +0.0001 | category counts by type and recency in-model | C02 arrays: nothing |
| 232 | per-user prior error / preference from earlier labelled feeds | user-specific bias toward age, popularity, similarity | label statistics (user) | scratchpad `ext/skeptic-uresid/`; logs only | earlier feeds; stale on test | -0.00005 .. -0.00027; split-half stability 0.03-0.18 | no | - |
| 233 | user-relative f4 / f5 profile | the user's preferred listing age and engagement level | user-item affinity | `ext/skeptic_userrel_attr/`; logs only | leak-free pre-test | +0.00325 on an f-only base, +0.00003 with popularity or pre2 | f columns + history | - |

### 2.4 The CCB batch (map l.258-271, 273)

Gate: nested LIN residual on G3 (user-grouped 5-fold), full-val weights applied to the holdout [R `nn_x/insights/ccb_1113/RESULT.md`]. All arrays V + H only unless noted.

| map l. | feature | information it carries | kind | built by; arrays | parity / leak status | gate on record (val; holdout) | trained equivalent today | lens today |
|---|---|---|---|---|---|---|---|---|
| 258 | C01 GBDT out-of-fold score + leaf-SVD64 | a LightGBM lambdarank on 134K exploration feeds | other (model output) | `ccb/tester1/out/c01_*` (score, n x 64 leaves) | train rows would need cross-fitting | -0.00264 +- 0.00064; -0.00136; score only -0.00008 +- 0.00021 | no | +0.4 / +0.4 / +1.9: no |
| 259 | C02 streak / EWMA (tau 4 / 16 / 64), strike, per-type same-category recency | recency-weighted similarity to the history | user-item affinity | `ccb/tester2/feats/c02_{val,hold}.npy` (n x 15) | own feed and history | -0.00084 +- 0.00041; +0.00002 | history encoder, similarity inputs | largest on val -2.9 (rec_e1; holdout -1.1): no |
| 260 | C03 latent geometry (loc16 subspace, mode 6 / 96, f2-supervised W) | location-like component of the content vector (also map l.199) | price / geo / category (latent) | `ccb/tester3/cache/c03_{val,hold}.npz` | label-free | -0.00008 +- 0.00030; -0.00029; g_cmatch +0.00022 +- 0.00014 | f2, content vector | at most 2.1 / 1.3: no |
| 261 | C04 co-visitation w5 / fwd5 / t1 / seqs + NB log-lift | windowed co-visitation variants | co-visitation | `ccb/tester1/out/c04_*` | day-causal source | -0.00020 +- 0.00034; -0.00030 | `seq_cv_*` | at most 2.5 / 1.3: no |
| 262 | C05 IDF / IUF / BM25 / idf-damped / Swing counts | re-weighted co-visitation counts | co-visitation | `ccb/tester2/feats/covis_{val,hold}.npy` (columns 0-4) | day-causal source | -0.00034 +- 0.00031; -0.00013 | `seq_cv_*` | cv_bm25 +2.6 / +2.6 / +3.8: weak |
| 263 | C06 Bradley-Terry item strength (`bt_theta`, `bt_seen`) | first-choice strength of the item among train feeds with p_random > 0.5 | label statistics (item) | `ccb/tester3/c06_build.py`; `cache/c06_{val,hold}.npz` | train labels only; holdout rows: one fit per day on earlier days; val: all train | +0.00028 +- 0.00037; -0.00015; it_tr_coec alone +0.00060 +- 0.00033 | no | **theta +5.4 / +5.3 / +6.9; seen -3.6 / -3.6 / -5.7: two witnesses** |
| 264 | C07 item trend / life cycle (`it_n7`, `it_trend`, `it_slope7`, `it_q95gap`, `exhaust`) | recent slope of the item's interactions at a gap-matched as-of | recency / trend | `ccb/tester1/out/c07_*`; V H Te | holdout as-of = feed day - 5 d (coverage 9.9% vs 23.1% on val) | +0.00031 +- 0.00038 (best of the batch); -0.00024 | no | at most 2.7 / 2.7, never both: no |
| 265 | C08 EMDE sketch (32 x 6 projections) | density of the history in random partitions of the content space | user-item affinity (content) | `ccb/tester2/feats/c08_*` (n x 6) | own history | -0.00063 +- 0.00036; -0.00055 | history encoder | at most 1.5 / 2.7: no |
| 266 | C09 RP3beta whole-sequence walk (`rp3b05`, `p3a`) | two-step walk candidate -> source sequences -> history items, degree-normalised, over WHOLE sequences | co-visitation | `ccb/tester3/c09_build.py` (560 s); `cache/c09_{val,hold}.npz` | source convention of `feat_covis.py` with earlier days only; same non-zero user skipped | +0.00012 +- 0.00036; -0.00013; cold-positive feeds -0.00400, warm +0.00211 | `seq_cv_*` use a +-20 position window; no whole-sequence or degree-normalised form | **+6.0 / +6.0 / +10.0: two witnesses, the strongest of all** |
| 267 | C10 user aggregates x 10 cues (160 interactions) | feed-constant user statistics as modifiers of cue weights | other (user segment) | interactions built in memory | holdout `tte_u_*` rebuilt as-of | -0.00155 +- 0.00058; -0.00141 | log H, user token | - |
| 268 | C11 typed implicit rating (`cvx_g1..g3`, intent, tw) | co-visitation split by the type of the user's history event | co-visitation | `covis_{val,hold}.npy` columns 5-9 | day-causal source | +0.00006 +- 0.00033; -0.00018 | `seq_cv_*`, per-type similarities | cvx_g3 +3.0 / +2.9 / +2.0: no |
| 269 | C12 post-processing key B (kappaB) | a metric-optimal re-ordering key from pair counts | other (function of the pool score; val-label constants) | `ccb/tester3/cache/c12_cols.npz` | uses val labels | -0.00006 +- 0.00032; +0.00003 | - | +4.4 / +4.6 / +5.0: not an input (a transform of the score) |
| 270 | C13 pending favourite p1 / p2 / p3 | items saved and not yet contacted | user-item affinity | `ccb/tester1/out/c13_*` | own history | -0.00010 +- 0.00017; -0.00006 | history tokens with event type | at most 1.2: no |
| 271 | C14 cross-feed prediction consensus | the pool's score of the item in other users' earlier val feeds | other (model output across feeds) | `ccb/tester2/feats/c14_*` | 0% coverage on the first val day, test never a source: no parity | STOP, coverage 4.54%; info +0.00022 +- 0.00018; -0.00050 | no | at most 2.3: no |
| 273 | C16 Sep-12 production-regime indicator (GBDT proxy) | change of the production ranker inside the train window | position / production prior | `ccb/tester1/out/c16_*` (model scores) | train-only concept | +0.00007 +- 0.00013; -0.00005 | domain token | - |

### 2.5 Later rows of T16 (map l.274, 277, 283, 305, 307) and l.276

| map l. | feature | information it carries | kind | built by; arrays | parity / leak status | gate on record | trained equivalent today | lens today |
|---|---|---|---|---|---|---|---|---|
| 274 | external-knowledge cues: seasonality per leaf, purchase-cycle class x recency, complement-category table, local hour, same-category counts by source, guest x category lift | sourced knowledge about categories | price / geo / category | `nn_x/insights/enrich_1113/`; `feats/{val,hold}.npz` | label-free | 0 of 9; best +0.00019 +- 0.00019 | category embedding | home flag -1.2 / -1.1 / -3.0 (one witness); the rest at most 1.8 |
| 276 | feed weekday + hour as inputs | time of the request | other (feed-constant) | `data/feats/tm_{hcos,hsin,weekend}` T V Te | fine | TRAINED 2-epoch screen: weekend slice +0.0002 +- 0.0005, overall 0.0000 | no | constant inside a feed |
| 277 | time-gap columns: `gap_last`, `gap_last_any`, `gap_first`, `never_seen`, `n_days_active` | when the item first / last appeared as a candidate in train feeds, on how many days | item exposure / life cycle | `nn_x/insights/timegap_1113/build_timegap.py` (23 s); `feats/{val,hold}.npz` | days before the row only; raw levels differ between train rows and val (PSI 6-9); within-feed z PSI 0.06-0.53 | 0 of 6; best gap_last +0.00030 +- 0.00027; gap_first holdout -0.00063 (z -3.05) | f4 (posting age) only | **gap_first -5.6 / -5.4 / -3.3; n_days_active -4.5 / -4.6 / -5.4; never_seen +2.2 / +1.8 / +3.9: two witnesses** |
| 283 | per-vertical experts / adapters | a separate scoring function per vertical | other (architecture) | CPU pre-gate, no array | - | per-vertical function minus global: -0.00095 +- 0.00039 | - | - |
| 305 | AUTOFE: 11,617 generated features, 19 blocks, OpenFE | transforms and pairs of the 56 inputs | feed composition (re-encodings) | `nn_x/w12/autofe/`; **33 recipe columns `afe_*` for T V Te** (float16) | label-free, constants frozen from val inputs | pass rates equal the noise controls; best nested block +0.00006 +- 0.00016 | derived from inputs | sd 1.26 / 1.23; largest 2.69 / 2.78; one of 33 at 2.5 on both (afe_grk_f5_rk -2.8 / -2.7): no |
| 307 | forensics nulls: earlier labelled feeds of the user, 246 columns as a block, zero-popularity pocket, seen-user gap | several | label statistics (user); other | `nn_x/w12/recheck/forensics_skeptic/`; `bank_{val,hold_g0,hold_g2}.npz` | earlier feeds only | (a) +0.00004 +- 0.00009; (b) +0.00016 +- 0.00029; nothing >= +0.0005 at 2 SE | no | see l.341 block a |

### 2.6 Other rows closed in residual or proxy form (found by the grep), and the feature store

| map l. | feature | information it carries | kind | built by; arrays | parity / leak status | gate on record | trained equivalent today | lens today |
|---|---|---|---|---|---|---|---|---|
| 143 | TRUNC-PROD | training-only rank-from-end offset on production rows | position / production prior (training offset) | `nn_x/w5` | - | GBDT proxy +0.0022 .. +0.0031 did NOT transfer: trained 10-epoch pairs -0.0005 +- 0.0005 val | - | - (a proxy that failed in the optimistic direction) |
| 149 | 2025-winners recipe columns: group-level EASE-DAN / RDLAE scorers, contact-ratio statistics, once-per-user counts, last-contact crosses | item-item linear auto-encoder scores; item / group / user click statistics | co-visitation; label statistics | `nn_x/w25/`; `data/{val,test}` 74 / 73 columns; no T, no H | statistics from all of train for val / test rows; no train-row build | column residuals on G3 <= +0.0006; member 0.3827 | partly | val only: `w25_it_dsf` -5.7; `w25_it_bctr2` +4.8; DAN / RDLAE scorers at most 3.6 |
| 159 | cross-user labelled kNN with an additive key | labels of similar users on the same item | label statistics (cross-user) | not built | - | closed by mechanism (key reduces to item-kNN + item prior) | no | - |
| 199 | location-subspace match of the content vector | region affinity | price / geo / category (latent) | `nn_x/emb/structure/` V 16; = C03 | label-free | within-feed AUC 0.536-0.551 | f2 | see C03 |
| 200 | s4 similar-listings item-to-item graph | "similar listings" clicks of other users | co-visitation | `nn_x/codes/followup/out/{val,hold}_s4_*.npy` | earlier days for holdout | coverage 0.05-0.13% of candidates; nested <= 0 | no | at most 1.1 / 2.2: no (coverage) |
| 224 | f5 forward echo | the item's f5 in other users' LATER feeds minus now | label statistics (leak class) | scratchpad `ext/russian/probe/` | uses later feeds; coverage depends on the position in the period | nested on G3 +0.00038 +- 0.00024 | no | NOT ADMISSIBLE |
| 227 | history extended from the user's earlier snapshots | events older than the 256-event cap | user-item affinity | `nn_x/w12/seg/`; V 60 arrays | earlier snapshots (allowed); gap 2.8 d on train, 14.2 d on test | replay +0.00117 +- 0.00082 / +0.00151 +- 0.00093; closed for lack of power | no | - (not a column) |
| 231 | seller-slot batch co-membership | items posted together | content | scratchpad `ext/xcheck193/` | - | coverage 0.1% | no | - |
| 234 | additive item CTR / COEC as a train-stage input (F17) | click rate of the item over the train window | label statistics (item) | `src/features_tte.py`: `tte_it_{imp,pos,ctr,coec}` T V Te; `xp_ctr_c` T V Te | train rows: days before the row (1 d fresh); val / test: all of train (2-4 d and 7-10 d stale): freshness defect | residual over base members about -0.0002; AUC 0.5225 -> 0.5013 from a 1-day to an 8-day gap; on G3 val +0.0004 .. +0.0006 +- 0.0004 at every gap | only inside the pop2 view (closed 11-13 / 11-15) | tte_it_coec +3.5 / +3.5 / +6.1; decays inside val (+3.6 -> +1.3) |
| 289 | pre2 encoder variants (19 stored) | the same encoder at other sizes | co-visitation (encoder) | `data/pretrain/feats_archive` V Te | - | slope of the nested residual +0.00003 per +0.001 zero-shot | `pre2_*` | - |
| 290 | hidden-event recall of the history encoder (SSL trigger) | self-supervised signal | other | `nn_x/w11/ssl/trigger/` | - | partial correlation with the G3 residual -0.0012 +- 0.0139 | - | - |
| 297 | hindsight targets from later snapshots | what the user did after the feed | label statistics (leak) | `nn_x/w12/hindsight/` | same user's later snapshots | +0.037 on covered feeds, a label copy | - | NOT ADMISSIBLE |
| 308 | production-selection (SELECT) score, item-CTR privileged teacher, co-click targets | which candidates production would select; item CTR as a teacher | position / production prior; label statistics | `nn_x/w12/recheck/winner_model`, `long_bets` | SELECT is a function of inputs | SELECT residual 0; PFD-item +0.0001; co-click GBDT proxy +0.00029 +- 0.00024 / +0.00003 | - | - |
| 327, 341 | CatBoost family B, four blocks: (a) the user's own earlier impressions (39 columns), (b) transition click rates (25), (c) nearest-train-item statistics (6), (d) near-duplicate statistics (11) | a: what the user was shown and clicked; b: P(click on key given history key); c, d: click and exposure statistics of the most similar much-shown train items, and the label-free density `kn_cos*`, `dup_n*` | label statistics (user, transition, neighbour); content density | `nn_x/w12/cb/famB/b3_build_{a,b,c,d}.py`; `feat/{a,b,c,d}_{xm,hold,val,test}.npy` (xm, H, V, Te; no full T); lag-matched `lag_{a,b,c}_{xm,hold}` | earlier feeds / other folds, earlier days; block a fresh on train (72 h) and stale on val / test (240 / 357 h); `kn_cos*` label-free with equal distributions on the four sets | nats next to [W16, tree]: all 51 click statistics +0.00004 +- 0.00007 / +0.00006 +- 0.00006; density pair +0.00018 +- 0.00015 / +0.00032 +- 0.00013 | no | **kn_cos1 -4.0 / -4.6 / -6.7 and 7 more density columns: two witnesses**; ua_item_imp -3.1 / -3.1 / -3.8; ue_k05_coec +2.6 / +2.6 / +3.7; tr_cat_base +3.2 / +2.5 / +2.7; kn_coec* at most 1.1 on val |
| 345 | fresh exposure counts from the val + test feeds (as-of and later-using windows) | appearances of the item in other users' val / test feeds | item exposure | `nn_x/w12/fresh/pregate/`; `out/cnt_main.npy` (4,294,777 x 6 = V + Te) | as-of form: density parity attainable; later-using forms depend on the position in the period | nested -0.00044 +- 0.00015 against a placebo of -0.00015 +- 0.00017; click lift given the pool 1.008 +- 0.021 | no | not read (sound closure: placebo + lift test) |
| 348 | neighbour-completed co-visitation for cold rows (3-step / 5-step walks) | co-visitation reached through content neighbours | co-visitation | `nn_x/w12/lit/int_gnn/walk_{all,train}_val.npy` (V x 3); critic `stageB/` | walk_all uses every history; walk_train train histories only; val only | cold rows: LR 4.40 (z form) / 11.37 (rank form), positive control 37.5; additive residual -0.00010 +- 0.00015 | no | 3-step walk s1, train source: +6.5 / +6.5 on ALL rows (val only); the same information as C09 (section 4.4) |
| 355 | output offset for rows deflated by the lag | calibration of stale popularity | item popularity (post-processing) | `reports/TRICKS_1115/web_shift_probe/` | - | nested -0.00012 +- 0.00031 despite a coefficient of +0.20 +- 0.03 | - | - (another case of likelihood without NDCG) |
| (78 file) | the feature store: 253 columns in `data/feats` and 31 in `data/feats_new` for T V Te, 38 more for V Te only | everything built in rounds 1-5 | all kinds | builders in `out/store_meta.tsv` | per column in `out/store_meta.tsv` (18 banned, 9 in-sample on train rows) | 11-11: 254 columns x 2 residual models on the nn_final4 mix: 3 / 2 nominal hits against 6 expected; best it_tr_coec GBM +0.00086 +- 0.00038 | 34 trained inputs, 63 in-model columns, 42 fine-tune extras | of 136 store columns that are not inputs, copies or banned: two-witness only `tte_it_coec`, `tte_it_ctr`, `tte_it_pos`, `xp_ctr_c`, `tte_ui_imp`, `tte_c_ctr`; `trd_*` (20) at most 2.5 / 2.6, `clu_*` (6) at most 2.3 / 3.5, unused `seq_*` at most 2.9 / 3.0, never both |

Rows that match the grep words and are NOT information rows (combiners, gates, losses, schedules), left out of the table: l.50, 87, 88, 90, 97, 100, 111, 120, 163, 219, 272, 280, 281, 302, 303, 343, 358.

## 3. How many reads, and what the best of that many noise reads looks like

Reads on record in residual or proxy form [I, from the reports' own counts]: feature store 254 columns x 2 models on the mix, 86 x 2 on the train-only half, 233 on the holdout, 22 group reads; the four new families about 20 screens; CCB 15 items and about 12 information columns; 94 DPA constructions and about 12 probes; 47 decoded cues; 9 enrichment cues; 6 time-gap items; 13 embedding reads; 17 pretrain2 reads; about 45 external-scan probes; item prior 5; AGE-PROD 6; external data 12; forensics 4 points, 12 arms, 248 columns; CatBoost family B 70 columns and 13 arms; FRESH and LIT about 40. Together about 1,400 hand-specified reads, plus the 23,234 generated tests of AUTOFE.

What pure noise gives in the project's nested form [M `out/noise_summary.txt`; typical maximum of N independent draws = the 1 - 1/(N+1) normal quantile]:

| N reads | typical best z | a one-column noise read on val (sd 0.00013) | a read with the usual SE of 0.0003 | on the holdout (noise sd 0.00005) |
|---|---|---|---|---|
| 16 (one CCB batch) | 1.56 | +0.00023 | +0.00047 | +0.00007 |
| 39 (the T16 rows) | 1.96 | +0.00028 | +0.00059 | +0.00009 |
| 94 (the DPA constructions) | 2.31 | +0.00033 | +0.00069 | +0.00011 |
| 254 (the feature store) | 2.66 | +0.00037 | +0.00080 | +0.00013 |
| 300 ("about 300 rows") | 2.71 | +0.00038 | +0.00081 | +0.00013 |
| 1,000 | 3.09 | +0.00043 | +0.00093 | +0.00015 |
| 11,617 (AUTOFE) | 3.76 | +0.00051 | +0.00113 | +0.00018 |

Measured directly: the best of the 300 placebo columns reads +0.00042 on val and +0.00019 on the holdout. So every single read quoted in `int_weak.md` T16 (+0.00060 +- 0.00033 at z 1.8, +0.00073 +- 0.00058 at z 1.3, +0.00045 +- 0.00022 at z 2.0, +0.00031 +- 0.00038) is inside what the best of its own batch of noise draws would print; none of them is support. Equally, none of the closures at "below +0.0005" is a closure: the same form reads real blocks worth +0.0013 at +0.0002 to +0.0009.

In my own sweep the multiplicity is: 305 candidate columns x 2 forms x 3 bases in the conditional test. Noise expects 3.8 columns at |cz| >= 2.5 on val, 3.5 on the holdout, and 0.02 on both with the same sign; observed 49, 54 and 25 [M `out/support_table.txt`].

## 4. Rows with support beyond a single in-sample read

### 4.1 Two witnesses, conditional on pool + the ranker's own inputs [M `out/support_table.txt`]

Same sign on val (train-only pool) and on the holdout, |cz| >= 2.5 on both in the z or the rank form. dnats = one-step loglik gain per feed. R2 = share of the column that pool + inputs reproduce on val.

| column | map l. | cz val TO (z / rank) | cz val FILE | cz holdout | R2 | dnats val / holdout | plain nested NDCG val TO / holdout (information only) |
|---|---|---|---|---|---|---|---|
| `c09_p3a` | 266 | +5.74 / +6.01 | +5.67 / +6.01 | +7.89 / +10.02 | 0.46 | 0.00120 / 0.00114 | -0.00001 +- 0.00043 / +0.00039 +- 0.00023 |
| `c09_rp3b05` | 266 | +5.84 / +5.97 | +5.85 / +6.00 | +8.20 / +9.75 | 0.41 | 0.00126 / 0.00122 | +0.00003 +- 0.00044 / +0.00041 +- 0.00023 |
| `c06_theta_l3` (Bradley-Terry) | 263 | +3.87 / +5.38 | +3.92 / +5.34 | +4.00 / +6.90 | 0.00 | 0.00052 / 0.00027 | +0.00031 +- 0.00028 / -0.00014 +- 0.00013 |
| `tg_n_days_active` | 277 | -4.53 / -3.92 | -4.60 / -3.85 | -5.11 / -5.42 | 0.62 | 0.00071 / 0.00043 | 0.00000 +- 0.00030 / -0.00022 +- 0.00011 |
| `kn_cos1` | 341 | -4.01 / -3.13 | -4.59 / -3.86 | -6.69 / -6.01 | 0.25 | 0.00057 / 0.00078 | +0.00006 +- 0.00014 / +0.00001 +- 0.00012 |
| `ex_n_s2` (leak-free) | 204 | -3.79 / -3.83 | -3.47 / -3.42 | -4.33 / -3.15 | 0.65 | 0.00049 / 0.00030 | -0.00020 +- 0.00027 / -0.00012 +- 0.00009 |
| `xp_ctr_c` | 234 | +3.67 / +3.29 | +3.59 / +3.07 | +5.75 / +5.77 | 0.02 | 0.00045 / 0.00054 | +0.00014 +- 0.00032 / +0.00007 +- 0.00016 |
| `c06_seen_l3` | 263 | -3.45 / -3.60 | -3.41 / -3.56 | -5.69 / -5.64 | 0.32 | 0.00040 / 0.00052 | -0.00030 +- 0.00027 / -0.00022 +- 0.00012 |
| `kn_cos2`, `kn_cos5`, `kn_cos10`, `kn_cos20`, `dup_n97`, `dup_ex97` | 341 | -3.52 .. -1.88 | -4.12 .. -2.67 | -6.56 .. -3.91 | 0.24-0.32 | | all within +-0.0002 |
| `tte_it_coec` | 234 | +3.48 / +3.46 | +3.49 / +3.33 | +6.14 / +6.10 | 0.02 | 0.00042 / 0.00063 | +0.00020 +- 0.00031 / +0.00006 +- 0.00017 |
| `ex_X3` (leak-free) | 204 | +4.54 / +3.34 | +4.17 / +3.09 | +3.08 / +3.29 | 0.48 | 0.00068 / 0.00016 | +0.00027 +- 0.00020 / +0.00019 +- 0.00011 |
| `tte_it_ctr`, `tte_it_pos` | 234 | +3.23 / +3.29; +3.55 / +2.46 | +3.13; +3.07 | +5.45 / +5.77; +3.25 / +2.80 | 0.04; 0.45 | | within +-0.0003 |
| `tg_gap_first` | 277 | -5.60 / -5.34 | -5.43 / -5.24 | -3.13 / -3.28 | 0.20 | 0.00110 / 0.00016 | -0.00007 +- 0.00039 / -0.00013 +- 0.00008 |
| `tte_ui_imp`, `ua_item_imp` | 234, 341 | -3.14; -3.07 | -3.18; -3.11 | -3.07; -3.79 | 0.00 | 0.00021 / 0.00013 | +0.00002 +- 0.00007 / +0.00001 +- 0.00003 |
| `tr_cat_base`, `tte_c_ctr`, `tr_c1024_base` | 341, store | +3.25; +2.74; +2.64 | +2.48; +2.07; +2.22 | +2.73; +2.66; +2.49 | 0.19-0.23 | | within +-0.0003 |
| `ue_k05_coec`, `ue_top5_pos` | 341 | +2.58; +2.56 | +2.63; +2.54 | +3.24 / +3.69; +2.38 / +2.66 | 0.01 | | within +-0.0002 |
| `cv_bm25` (covis col 2) | 262 | +2.58 / +2.13 | +2.58 | +2.95 / +3.83 | 0.91 | | within +-0.0003 |
| `c12_kappaB` | 269 | +4.36 | +4.60 | +5.00 | 0.73 | | a transform of the pool score; not an input |

One witness only: `ixh_mf1` (-4.3 val, -0.3 holdout), `ixh_ln` / `tte_it_imp` / `xp_imp_c50` (-1.0 .. -1.8 val, -4.4 .. -4.7 holdout: an exposure count, a popularity-type contrast that only the fresh holdout shows), `dup_ex95`, `pre2n_hmax32`, `X2`, `gap_first_any`.

Val-only arrays [M `out/valonly.txt`, `out/support_table.txt`]: `walk_all_c0` +7.2 / +7.7, `walk_train_c0` +6.3 / +6.5 (the graph walks of l.348); `als_exp5_rk` +5.1, `als_exp5` +4.0 / +4.4, `als_last10` +4.0 / +4.4 (implicit-ALS profile cosines, val / test only); `ip_x_val_T` +4.5 / +4.9; `w25_it_dsf` -5.7, `w25_it_bctr2` +3.5 / +4.8; `emb_chg_d` +2.8 / +3.2; `ud_cos_prev` +2.6 / +2.7. Of 94 val-only columns, 17 read |cz| >= 3 (noise expects 0.3).

### 4.2 Where the information sits [M `out/slices.txt`]

cz of the z form on label-free feed slices. The weighted columns use the feed weights of `src/pmix.py`; weights lower the effective sample, so compare signs and ratios inside a row.

| column | coverage val / holdout | val all | val day 20718 (2 d after train) | val day 20720 (4 d) | cold-share terciles of the feed: low / mid / high | pmix-public weights | pmix-private weights | holdout Sep 6-8 / 9-16 / 17-19 |
|---|---|---|---|---|---|---|---|---|
| `c09_rp3b05` | 0.198 / 0.156 | +5.84 | +3.81 | +4.46 | +4.65 / +3.21 / +2.07 | +4.20 | +3.36 | +2.57 / +7.09 / +3.60 |
| `walk_train_c0` (val only) | 0.212 | +6.28 | +4.27 | +4.64 | +5.13 / +2.94 / +2.67 | +4.31 | +3.34 | - |
| `kn_cos1` | 0.999 / 0.999 | -4.01 | -1.82 | -3.89 | -2.61 / -2.79 / -1.54 | -2.13 | -1.27 | -3.63 / -5.11 / -2.56 |
| `c06_theta_l3` | 0.439 / 0.386 | +3.87 | +3.00 | +2.46 | +2.84 / +0.61 / +3.23 | +3.49 | +3.25 | +1.76 / +3.31 / +1.60 |
| `ip_x_val_T` (val only) | 0.464 | +4.47 | +3.10 | +3.21 | +3.06 / +0.39 / +4.24 | +3.59 | +3.18 | - |
| `tte_it_coec` | 0.654 / 0.644 | +3.48 | +3.63 | +1.26 | +2.28 / +2.38 / +1.39 | +1.73 | +1.15 | +3.13 / +3.40 / +4.28 |
| `xp_ctr_c` | 0.654 / 1.000 | +3.67 | +3.86 | +1.31 | +2.05 / +2.65 / +1.68 | +1.97 | +1.35 | +3.38 / +3.06 / +3.80 |
| `tg_gap_first` | 0.651 / 0.640 | -5.60 | -3.88 | -4.04 | -2.65 / -3.65 / -3.43 | -4.33 | -3.87 | +0.08 / -1.72 / -3.31 |
| `tg_n_days_active` | 0.651 / 0.640 | -4.53 | -4.22 | -2.12 | -3.94 / -3.76 / -0.18 | -0.99 | +0.32 | -1.85 / -3.31 / -3.66 |
| `ex_X3` (leak-free) | 0.508 / 0.457 | +4.54 | +3.20 | +3.25 | +2.62 / +2.10 / +3.10 | +4.17 | +3.91 | -0.30 / +2.70 / +2.38 |
| `ex_n_s2` (leak-free) | 0.358 / 0.280 | -3.79 | -2.51 | -2.89 | -1.88 / -1.47 / -3.18 | -3.39 | -3.15 | -1.14 / -3.09 / -3.18 |
| `tte_ui_imp` | 0.002 / 0.002 | -3.14 | -3.09 | -1.23 | -0.81 / -2.59 / -2.22 | -2.87 | -2.05 | -0.39 / -2.24 / -2.18 |
| `tr_cat_base` | 1.000 / 0.910 | +3.25 | +3.41 | +1.16 | +2.36 / +2.20 / +1.10 | +2.10 | +1.54 | +1.61 / +1.40 / +1.87 |
| `ixh_mf1` | 0.654 / 0.644 | -4.32 | -2.14 | -4.02 | -1.58 / -3.35 / -2.54 | -4.09 | -3.95 | +1.47 / +0.33 / -2.09 |

Readings.
- The co-visitation walk, the surface split, the exploration-feed click strength and gap_first keep their sign and most of their size from the first to the second val day and under the private weights.
- The all-feed click statistics (`tte_it_coec`, `xp_ctr_c`), the category base rate, `n_days_active` and `tte_ui_imp` lose two thirds between a 2-day and a 4-day gap (val days differ also by weekday; this is the only gap contrast val offers). Test rows are 6.6 to 9.6 days out. This agrees with the record (AUC 0.5225 -> 0.5013 from a 1-day to an 8-day gap, `ext/skeptic_f17/decay.log`). They are freshness signals: a defect, not a lever.
- The walk is defined on warm items only (non-zero on 19.8% of val rows, 15.6% of holdout rows) and weakens with the cold share of the feed (+4.65 -> +2.07); the density column is defined on every row.

### 4.3 Groups, three reads each, next to positive controls read the same way [M `out/s07_shortlist.log`, `out/shortlist.json`]

(a) LR of the group (z and rank form of each column) given pool + the ranker's own inputs; (b) nested NDCG of pool + group; (c) cross-witness: coefficients of [pool, group] fitted on one witness, applied unchanged to the other.

| group | columns | (a) conditional LR val TO / val FILE / holdout (df) | nats per feed val / holdout | (b) nested NDCG val (pmix-private) / holdout | (c) holdout coefficients on val (pmix-private) | (c) val coefficients on the holdout |
|---|---|---|---|---|---|---|
| G1 co-visitation walk | p3a, rp3b05 | 50.5 / 50.4 / 132.7 (4) | 0.00175 / 0.00221 | -0.00020 +- 0.00046 (-0.00126) / +0.00055 +- 0.00025 | +0.00001 +- 0.00046 (-0.00077) | +0.00029 +- 0.00025 |
| G2 catalogue density | kn_cos1, kn_cos20 | 17.1 / 21.4 / 48.2 (4) | 0.00059 / 0.00080 | +0.00028 +- 0.00023 (+0.00033) / +0.00006 +- 0.00013 | +0.00024 +- 0.00022 (+0.00039) | +0.00009 +- 0.00013 |
| G3 item click statistics | tte_it_coec, tte_it_ctr, bt_theta | 33.3 / 33.0 / 72.7 (6) | 0.00115 / 0.00121 | +0.00020 +- 0.00040 (+0.00015) / -0.00008 +- 0.00020 | +0.00029 +- 0.00036 (-0.00004) | -0.00001 +- 0.00022 |
| G4 item life cycle | n_days_active, gap_first, never_seen | 47.5 / 42.8 / 33.8 (6) | 0.00165 / 0.00056 | -0.00024 +- 0.00042 (-0.00012) / -0.00019 +- 0.00013 | +0.00017 +- 0.00027 (+0.00026) | -0.00061 +- 0.00023 |
| G5 surface split | X3, n_s2 (leak-free) | 23.5 / 19.5 / 23.7 (4) | 0.00081 / 0.00040 | +0.00022 +- 0.00032 (+0.00030) / -0.00004 +- 0.00013 | +0.00005 +- 0.00023 (+0.00003) | -0.00034 +- 0.00017 |
| G6 the user's earlier feeds | tte_ui_imp, ue_k05_coec, ue_k20_coec | 16.8 / 17.2 / 24.8 (6) | 0.00058 / 0.00041 | +0.00007 +- 0.00021 (-0.00025) / -0.00019 +- 0.00010 | +0.00006 +- 0.00019 (-0.00012) | -0.00017 +- 0.00011 |
| G7 category base rate | tr_cat_base, tte_c_ctr | 14.0 / 8.6 / 9.3 (4) | 0.00048 / 0.00016 | +0.00004 +- 0.00027 (-0.00014) / -0.00005 +- 0.00009 | +0.00021 +- 0.00021 (+0.00015) | +0.00001 +- 0.00012 |
| G8 exposure count | tte_it_imp | 3.1 / 3.0 / 22.2 (2) | 0.00011 / 0.00037 | +0.00006 +- 0.00024 (-0.00024) / -0.00012 +- 0.00011 | -0.00009 +- 0.00019 (-0.00022) | -0.00009 +- 0.00012 |
| B1 bundle G1 + G2 | 4 columns | 66.9 / 71.0 / 178.3 (8) | 0.00232 / 0.00297 | -0.00013 +- 0.00049 (-0.00048) / +0.00068 +- 0.00027 | -0.00011 +- 0.00048 (-0.00097) | +0.00059 +- 0.00027 |
| B2 bundle G1 + G2 + G3 + G4 | 8 columns | 130.9 / 132.8 / 262.3 (16) | 0.00454 / 0.00437 | +0.00022 +- 0.00060 (-0.00025) / +0.00051 +- 0.00030 | +0.00067 +- 0.00054 (+0.00017) | +0.00010 +- 0.00033 |
| R2 = ranks 1-3 of section 5 | p3a, rp3b05, kn_cos1, kn_cos20, X3, n_s2 | 88.4 / 88.5 / 198.7 (12) | 0.00306 / 0.00331 | +0.00032 +- 0.00053 (+0.00036) / +0.00046 +- 0.00029 | +0.00045 +- 0.00050 (+0.00004) | +0.00009 +- 0.00030 |
| R1 = ranks 1-5 of section 5 | R2 + bt_theta, gap_first, never_seen | 150.4 / 146.7 / 258.9 (18) | 0.00521 / 0.00432 | -0.00005 +- 0.00060 (-0.00024) / +0.00039 +- 0.00030 | +0.00112 +- 0.00055 (+0.00083); on the val FILE +0.00047 +- 0.00054 | +0.00022 +- 0.00034 |
| N1 negative control: inputs the ranker reads | seq_cv_invd20, it_hist_all_luo, pre2_fe_z | 0.0 / 0.0 / 0.0 (6) | 0 / 0 | +0.00002 +- 0.00038 (-0.00019) / +0.00006 +- 0.00017 | -0.00003 +- 0.00032 (-0.00011) | +0.00019 +- 0.00021 |
| P1 positive control: 8 popularity inputs on the popularity-light wide students (true +0.00134 +- 0.00090 / +0.00636 +- 0.00061) | 8 columns | 168.4 / - / 837.9 (16) | 0.00584 / 0.01397 | +0.00040 +- 0.00065 (-0.00077) / +0.00258 +- 0.00042 | +0.00056 +- 0.00069 (-0.00032) | +0.00258 +- 0.00043 |
| P2 positive control: encoder scalars on the members without them (true +0.00061 +- 0.00083 / +0.00059 +- 0.00049) | 3 columns | 63.5 / - / 88.0 (6) | 0.00220 / 0.00147 | +0.00037 +- 0.00051 (+0.00026) / +0.00035 +- 0.00023 | +0.00041 +- 0.00044 (+0.00032) | -0.00005 +- 0.00025 |

Readings.
- Positive controls: the cross-witness read recovers 0.42 (val) and 0.41 (holdout) of the popularity block and 0.66 / -0.08 of the encoder block; with SEs of 0.0004 to 0.0007 on val it cannot tell a block worth +0.0013 from zero. The same reads of the candidate bundles are therefore neither confirmation nor rejection.
- The one read above 2 SE is R1, holdout coefficients on val: +0.00112 +- 0.00055. Its composition was chosen from the single-column results on the same two witnesses, and it is the largest of 13 group reads: information, not evidence.
- At the private mix the walk alone reads negative in nested NDCG (-0.00126; SE about 0.0007) while its conditional score stays positive under the same weights (+3.4, section 4.2): the cold / warm trade of a residual on a finished pool, as the popularity control shows it too (P1 nested at the private mix -0.00077 for a block that is worth +0.0013).
- Information per feed (nats): G1 alone is 0.8 (val) / 1.5 (holdout) of the encoder block; the 9-column bundle 2.4 / 2.9 of it, and 0.9 / 0.3 of the popularity block.
- G4 and G5 read negative on the holdout with val-fitted coefficients (-0.00061 +- 0.00023, -0.00034 +- 0.00017): their best weights differ between a 1-day and a 2-4-day gap. They are item-history columns and enter, if at all, in a within-feed form with the lagged twin as the witness.

### 4.4 Are the families the same information? [M `out/overlap.txt`, val, train-only pool]

cz (z form / rank form) of a column given pool + the ranker's own inputs, and given that plus the z and rank forms of one other family.

| column | given inputs | + RP3 (p3a, rp3b05) | + graph walk (2 columns) | + ALS (3 columns) | + density | + bt_theta | + surface split |
|---|---|---|---|---|---|---|---|
| `c09_rp3b05` | +5.84 / +5.97 | in the block | +1.00 / +0.60 | +5.55 / +5.49 | +5.79 / +5.95 | +5.87 / +6.03 | +5.76 / +5.88 |
| `walk_train_c0` | +6.28 / +6.48 | +2.40 / +2.41 | in the block | +6.02 / +5.97 | +6.21 / +6.44 | +6.28 / +6.54 | +6.20 / +6.38 |
| `als_exp5_rk` | +5.13 / +5.13 | +4.61 / +4.61 | +4.89 / +4.88 | in the block | +5.20 / +5.20 | +5.19 / +5.19 | +5.11 / +5.11 |
| `kn_cos1` | -4.01 / -3.13 | -3.92 / -3.08 | -3.87 / -3.04 | -4.11 / -3.24 | in the block | -3.88 / -3.01 | -3.97 / -3.08 |
| `c06_theta_l3` | +3.87 / +5.38 | +3.92 / +5.42 | +3.90 / +5.41 | +3.86 / +5.44 | +3.85 / +5.28 | in the block | +3.78 / +5.29 |
| `ip_x_val_T` | +4.47 / +4.90 | +4.44 / +4.91 | +4.41 / +4.90 | +4.43 / +4.95 | +4.42 / +4.81 | +0.56 / +0.22 | +4.32 / +4.72 |
| `ex_X3` (leak-free) | +4.54 / +3.34 | +4.20 / +2.93 | +4.18 / +2.92 | +4.42 / +3.17 | +4.52 / +3.36 | +4.53 / +3.48 | in the block |
| `tg_gap_first` | -5.60 / -5.34 | -5.59 / -5.32 | -5.66 / -5.39 | -5.36 / -5.08 | -5.66 / -5.38 | -5.51 / -5.22 | -5.30 / -5.07 |

The RP3beta walk and the graph walk are one piece of information; x_T and the Bradley-Terry strength are one piece; ALS, density, surface split and first-seen gap are separate pieces. Six independent pieces in all, five of them cheap to build.

### 4.5 Robustness to a richer null model, and where the walk adds [M `out/robust.txt`]

Null B = pool + the ranker's own inputs + 20 non-linear forms of the collaborative inputs (rank forms and squared z of all 8 popularity / co-visitation inputs, the non-zero flag and quantile transforms of seq_cv_invd20 / seq_cv_norm, the products seq_cv_invd20 x it_hist_all_luo and seq_cv_invd20 x seq_max_all).

| column | val: cz given own inputs (z / rank) | val: given null B | holdout: given own inputs | holdout: given null B |
|---|---|---|---|---|
| `c09_rp3b05` | +5.84 / +5.97 | +5.83 / +6.17 | +8.20 / +9.75 | +8.32 / +9.92 |
| `kn_cos1` | -4.01 / -3.13 | -4.12 / -3.24 | -6.69 / -6.01 | -6.75 / -6.09 |
| `ex_X3` (leak-free) | +4.54 / +3.34 | +4.65 / +3.61 | +3.08 / +3.29 | +3.34 / +3.44 |
| `c06_theta_l3` | +3.87 / +5.38 | +3.80 / +5.14 | +4.00 / +6.90 | +4.03 / +6.97 |
| `tg_gap_first` | -5.60 / -5.34 | -5.50 / -5.24 | -3.13 / -3.28 | -3.16 / -3.30 |

No column moves: the information is not a non-linear re-reading of the collaborative inputs the ranker has.

Where the walk adds: it is non-zero on 19.8% of val rows (15.6% of holdout rows), the windowed input `seq_cv_invd20` on 11.3% (8.5%); the walk is non-zero on 8.5% (7.1%) of rows where the windowed input is zero and never the reverse. Split by that, the walk reads +4.2 on rows WITHOUT a windowed hit and +4.1 on rows with one (holdout +7.3 / +4.6). It both widens the collaborative coverage and refines it.

### 4.6 Mechanism: can the ranker derive the column from what it reads? [I, from `docs/02_SOLUTION.md` 4.1-4.2 and the builders]

| column | what would be needed to derive it | derivable from present inputs? |
|---|---|---|
| RP3beta walk | the cross-user source index over whole sequences and item degrees | no: `seq_cv_*` are sums inside a +-20 position window without degree normalisation (non-zero on 11.3% of val rows against 19.8% for the walk); `pre2_*` see the user's own trajectory through an encoder with an ID table; R2 with pool + inputs 0.41-0.46; unchanged under null B of 4.5 |
| kn_cos (density) | a catalogue-wide index of much-shown items | no: `ctx_nd_*` measure neighbour density among the candidates of the feed only; R2 0.25 |
| X3, n_s2 (surface split) | counts of card views of the item by OTHER users, split by the surface they came from | no: `it_hist_et1_luo` is the total; the source code is an input only for the user's own history events; R2 0.48-0.65 |
| bt_theta / x_T (exploration-feed click strength) | train labels of other users on random-like feeds | no: no label statistic is an input; R2 0.00-0.05 |
| gap_first | the first day the production system offered the item | partly: f4 gives the posting age; conditional on f4 it still reads -5.6 / -3.3; R2 0.20 |
| all-feed item CTR, category base rate | train labels, mostly of production feeds | no, but the content decays in days |
| tte_ui_imp, ue_k* | the user's earlier labelled feeds | no; coverage 0.2% (same item) to 31% (kernel statistics) of rows, sources 10-15 days old on val / test [R `reports/CATBOOST_1115/famB.md` 2.2] |

## 5. Short list and NOT ADMISSIBLE

### 5.1 Candidates for a trained-input bundle, ranked (at most 15)

Rank = two-witness support, then stability under staleness, then parity risk and cost. "On disk" = arrays with parity today. CPU cost = wall time at 12 threads for train + holdout (inside train) + val + test, from the build logs [R] scaled by row or feed count [I].

| rank | columns | support (cz val TO / val FILE / holdout; stability) | on disk | what a parity build needs | CPU cost | risk |
|---|---|---|---|---|---|---|
| 1 | `rp3b05`, `p3a` (RP3beta whole-sequence walk) | +6.0 / +6.0 / +10.0; stable over val days and at the private weights; nested holdout +0.00055 +- 0.00025, val -0.00020 +- 0.00046 | V, H (`ccb/tester3/cache/c09_*.npz`) | run `c09_build.py` logic with the EXACT source convention of the adopted `feats_v2/seq_/feat_covis.py` (train row of day d: train source before day d; val / test: all train + earlier val / test histories; same non-zero user skipped) for T and Te; a lagged twin for the stale witness as in the hz2 view | 1.5 h (560 s for val + 30,000 holdout feeds with 14 day indexes [R `c09_build.log`]; queries scale by 105) + 0.5 h twin | stale on test like `seq_cv_*`; non-zero on 20% of rows; cold feeds gain least; cold / warm trade on record (-0.0040 / +0.0021) |
| 2 | `kn_cos1`, `kn_cos20` (cosine to the nearest much-shown train items of the candidate's 4096-cluster) | -4.0 / -4.6 / -6.7; defined on every row; cross-witness +0.00024 +- 0.00022 val | xm, H, V, Te (`nn_x/w12/cb/famB/feat/c_*.npy`, columns 4 and 5) | run `b3_build_c.py` for all train rows; index = items with >= 5 impressions (decide: with the holdout feeds for all-data members); cluster ids exist for T V Te (`data/clu_work/cand_cl_4096_*.npy`) | 0.5 h (191 s for 28.7M rows [R famB 2.2]) | index membership uses all 15 train days, so a train row sees slightly more catalogue than a test row; distributions equal on the four sets [R `famB_review.md` l.39] |
| 3 | `X3 = log((n_s1+1)/(n_s2+1))`, `n_s2`, leak-free | +4.5 / +4.2 / +3.3 and -3.8 / -3.5 / -4.3; stable at the private weights | V, H (`nn_x/codes/exploit/02_s1s2contrast/feats_*.npz`, keys `val_lf_*`) | counts of distinct other users with a feed-opened / search-opened card view, from snapshots of strictly earlier days; build T and Te with the day rule of rank 1; same-user and user-0 handling as LUO | 1 h (194 s for val + holdout [R `run.log`]) | item-history signal: stale on test; never use the "V-exact" variant (forward leak, measured) |
| 4 | `bt_theta` (Bradley-Terry strength on train feeds with p_random > 0.5) or the simpler prior x_T | +5.4 / +5.3 / +6.9; x_T +4.9 on val; no decay between the val days | V, H (`ccb/tester3/cache/c06_*.npz`); x_T: V, Te (`nn_x/audit/itemprior/`) | per-day fits exist for the 14 train days (rows of day d read the fit on earlier days); test = the all-train fit; lag-matched twin; the 30,000 holdout feeds as label sources only for all-data members | 0.3 h (190 s incl. 14 fits [R `c06_build.log`]) | a label statistic: must stay causal for train rows; covers 44% of val rows, 37-40% of test rows [R itemprior out.txt] |
| 5 | `gap_first` (days since the item first appeared as a train candidate), within-feed rank form, + `never_seen` flag | -5.6 / -5.4 / -3.3; stable at the private weights; on the holdout only in the later days (+0.1 / -1.7 / -3.3 for Sep 6-8 / 9-16 / 17-19: early rows have no look-back) | V, H (`nn_x/insights/timegap_1113/feats/*.npz`) | day masks per item (`build_timegap.py`), all splits; ONLY the within-feed rank / z may enter (raw PSI train -> val 6-9) | 0.2 h (23 s for a 300K-feed sample + V + H) | level parity is structurally impossible (train rows look back 0-13 d, test rows 7-23 d); partly f4 |
| 6 | day-causal implicit ALS profile cosines (`als_exp5`, rank form) | +4.0 .. +5.1 on val (both pools), val only; +4.6 after rank 1 is in the null model [M `out/overlap.txt`]: separate information | V, Te (`data/feats/{val,test}/als_*`): factors fitted on train + val + test histories, usable for val / test rows only | a causal build: factors from histories before day d for train rows of day d (14 fits), the all-train fit for val / test; same-user and user-0 rule of LUO | 4-8 h on CPU [I] (the stored factors came from a GPU solver, `src/features_als.py`) | one witness; "time-windowed ALS" is on the do-not list of 11-11 on a GBDT read; NOT admissible as built |
| 7 | graph walk of 11-15 (`walk_train`) | +6.3 / +6.5 on val, val only; +2.4 after rank 1 | V | as rank 1 with the content-neighbour step; 5 CPU-h + parity [R `reports/LIT_1115/CRITIC.md` section 4] | 5 h | the same information as rank 1 (rank 1 reads +0.4 .. +1.0 after it): do not build both |
| 8 | `tte_ui_imp` (same user was shown this item in a train feed) | -3.1 / -3.2 / -3.1 | T V Te (`data/feats`) | on disk, but fresh on train and stale on test (5.2): a lag-matched rebuild would be needed | 0 as built | non-zero on 0.2% of rows: no pool-level value; listed for completeness, not for the bundle |
| 9 | `seq_cos_last1` (cosine to the newest history event) | +2.3 / +2.3 / +3.0 | T V Te (`data/feats`) | on disk, own history only, no parity issue | 0 | weak; the history encoder sees the newest event |
| 10 | `cv_bm25` (BM25-weighted co-visitation) | +2.6 / +2.6 / +3.8 | V, H | as rank 1 | with rank 1 | weak; R2 0.91 with pool + inputs |
| 11 | `afe_grk_f5_rk` (rank of f5 inside (feed, category)) | -2.8 on val (train-only pool), -2.7 on the holdout | T V Te (`nn_x/w12/autofe/feats`) | on disk | 0 | one of 33 generated columns; a re-encoding |
| 12 | `ixh_mf1` | -4.3 / -4.2 on val, -0.3 on the holdout | T V Te (`data/feats_new`) | on disk; NaN-share defect (64% train, 35-47% val / test) | 0 | one witness; not recommended |

Recommended bundle [I]: ranks 1-5 (9 columns: p3a, rp3b05, kn_cos1, kn_cos20, X3, n_s2, bt_theta, gap_first as rank, never_seen; read jointly as R1 in section 4.3), built once for all four splits, entered through `XT_PRESETS` like the `afe` recipe of `reports/AUTOFE_1114.md` section 7, the item-history columns as within-feed rank / z plus a has-flag. Build 3.5 h + twin 0.5 h; parity checks before any GPU: train(random-like, M >= 50) vs val PSI below 0.5 on the forms that enter, zero share and zero-lift by day, the young-listing table of `reports/featrev_prep.md` s14, the censored-zero lift of `reports/featnew_hzcv.md` section 3, and a label-free count of test rows with a non-default value by test day. Half a day to a day.

Before any GPU, two CPU reads decide what enters (each minutes with `condlib.py`): (i) the conditional test of every built column on stale-val inputs (the lagged twin), which the fresh val here cannot give; (ii) a label-free count of non-default rows by test day (drop a column whose share on private-day rows is below half of val's). The overlap between ranks 1, 6 and 7 is already read (section 4.4).

### 5.2 NOT ADMISSIBLE

| column / row | reason |
|---|---|
| `it_hist_all`, `it_hist_et*` without `_luo` (14 columns) | same user's later snapshots; cz +6.9 on val, +40.5 on the holdout |
| `it_cand_cnt_split*`, `it_cand_cnt_train` | in-split candidate counts include the same user's later feeds; train rows count themselves; cz +17.8 on the holdout |
| `it_tr_*`, `u_tr_*` | train rows are in-sample (own label only left out by row): cz +8.7 .. +10.8 on the holdout; the causal form is `tte_*` |
| train-row `pre_*`, `x_lh_*` (quarantine) | content-table membership leak; later-history signal (map l.77, 123) |
| f5 forward echo (l.224), hindsight targets (l.297), later-using exposure windows (l.345), later-using co-visitation, the "V-exact" surface counts of l.204 | read feeds or snapshots LATER than the row; coverage depends on the position of the row in the period (a val row has all test days ahead, a late private row none), so train, val and test rows cannot be built alike; l.297 is the same user's later snapshot |
| val-label priors: item prior V of l.140, C12 key constants, any `*_oof` statistic of val labels | val labels are admissible only inside nested user-grouped folds, i.e. at the fine-tune stage; no train-row analogue exists |
| lifelong history from earlier snapshots (l.227); `ud_*` (l.78); the user's earlier labelled feeds: block a of l.341, l.226, l.232, `tte_u_*`, `tte_uc_*`, `tte_ui_*` | allowed by the leak rule (earlier snapshots / feeds of the same user), but fresh on train rows and stale on test rows: previous feed 72 h back on train against 240 h on val and 357 h on test; earlier snapshot 2.8 d against 14.2 d; under the test lag 5.4% of training feeds keep a source [R famB.md 2.2, 0.4]. Admissible only in a lag-matched build, which leaves too few rows; user 0 is never linked |
| all-feed item click statistics as built: `tte_it_ctr`, `tte_it_coec`, `tte_it_pos`, `xp_ctr_c`, `w25_it_*`, `tr_cat_base`, `tte_c_ctr`; `n_days_active` | train rows read the day before, test rows read 7-10 days back; the decay is measured twice (AUC 0.5225 -> 0.5013 over 8 days [R]; cz +3.6 -> +1.3 between the two val days [M]). The trained form (pop2 view) is closed |
| `emb_chg_d`, `chg_flag`, `in_train` (l.198) | need the item table of a later split: no train-row analogue |
| `als_*` as built | factors fitted on train + val + test histories with the users' later snapshots inside: valid for val / test rows only (`src/features_als.py` header); the admissible form is rank 6 of 5.1 |
| C14 cross-feed consensus (l.271) | the source is other users' earlier val feeds: 0% on the first val day, never on test |
| C01 score / leaves, SELECT score, tree scores as inputs (l.245, 258, 308) | label-trained model outputs: train rows need cross-fitting; and the lens reads no information in them (+0.4 / +1.9) |
| `ixh_*` under the shared lag | not a leak, but 64% NaN on train against 35-47% on val / test; one witness |

## 6. Limits

- No trained arm was run; nothing here prices a column in NDCG. The group reads of 4.3 show that the residual NDCG form cannot.
- The conditional lens is linear in two forms of a column. Interactions (a column that matters only for cold rows, or only through the history) are not seen. The control block is 84 columns, not the model's full function.
- The holdout is in-period with fresh item histories; for G1, G3, G4, G5 it is the optimistic witness. Val is 2-4 days out; test is 6.6-9.6 days out. No stale-val read was made (needs lagged builds).
- Holdout rows of gate arrays are assumed to be in the order of `nn_x/w5/preds/holdout/holdout_meta.npy` (all gates used copies of one harness); a misaligned array would read zero, not a signal.
- The two val days differ by weekday as well as by gap.
- Not read: the 94 DPA constructions, the P3 arrays (2-D), the FRESH counts (closed with a placebo), C10 interactions, `kv64_*` (feed-level shapes).
- The counts of reads on record in section 3 are taken from the reports' own statements, not recounted from logs.

## 7. Files

All under `reports/INPUTS_1116/residual_rows/`.
- Scripts: `s01_disk_scan.py`, `lens.py`, `s03_calibrate.py`, `cand_specs.py`, `s04_sweep.py`, `s05_meta.py`, `s06_cond.py`, `condlib.py`, `s07_shortlist.py`, `s07b_bundle.py`, `s08_tables.py`, `s09_slices.py`, `s10_afe.py`, `s11_valonly.py`, `s12_overlap.py`, `s13_robust.py`, `run_sweeps.sh`.
- Outputs in `out/`: `disk_scan.tsv`; `calib_{val,hold}.json` + logs; `sweep_{val_TO16,val_FILE,hold_TO16}.jsonl`; `cond_{val_TO16,val_FILE,hold_TO16}.{jsonl,json}` + logs; `store_meta.tsv`; `all_columns.tsv` (every column, every read); `support_table.txt`; `noise_summary.txt`; `negctl_existing_inputs.txt`; `shortlist.json`, `s07_shortlist.log`; `shortlist_R.json`, `s07b_bundle.log`; `slices.txt`; `afe.txt`; `valonly.txt`; `overlap.txt`; `robust.txt`.
