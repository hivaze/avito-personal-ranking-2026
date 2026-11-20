# INVENTORY C: the feature blocks of the CatBoost rounds and of the 2025-winners recipe as candidates for TRAINED inputs of the neural ranker

Written 2026-11-16 02:45 UTC. CPU only (12 threads per process, at most two processes of mine at a time, no GPU, no process signalled). Read-only on the project; scripts and outputs are in `reports/INPUTS_1116/catboost_blocks/`. Nothing was trained: every number of mine is an evaluation read on stored predictions and stored feature arrays.

Tags: **[M]** measured here (script and output file named; outputs are under `reports/INPUTS_1116/catboost_blocks/out/`), **[R]** record on disk (path named), **[I]** judgement. "a ± b" = mean ± 1 SE, paired per feed. "x / y" = val (14,423 feeds) / holdout (30,000 in-period feeds) unless the cell says otherwise.
Names: W16 = the train-only pool of 16 wide students that the CatBoost rounds used as "the pool" (0.40144 / 0.32460). B0 = CPU YetiRank on the 177 tree features (0.38150 / 0.30462). B3 = B0 + the 70 columns of family B. "Full view" = the 34 pre-computed inputs + 63 in-model columns of the wide students (docs/02_SOLUTION.md section 4).

## 0. The answer

### 0.1 In five lines

1. **22 blocks were inventoried** (605 hand-made columns on disk, 4 constructions that live only inside CatBoost models, the 12 `tte_*` columns of the tree base, tree scores). **No block passes all three tests (a), (b), (c) as it stands on disk.** Three can be made to pass with a rebuild, and a fourth entry is a carrier for them; they are ranked in 0.2.
2. **Nothing in this inventory is of the size of the gap to #1.** Next to W16 the ranks 1 and 2 together (three columns) read +0.0008 nats on both witnesses [M], which the calibration of section 1 maps to about +0.0013 NDCG as trained inputs on fresh rows (range +0.0006 to +0.004), and to less on rows as stale as test. Reweighted to the private feed mix the val read is +0.00006 ± 0.00047 [M].
3. **The one block with a real case is the item click rate from train feeds** (`tte_it_ctr` / `tte_it_coec`, NN-ready as `xp_ctr_c`). The full view reads no click statistic; the column was a trained input only in the popularity-free view (10 runs, 0 on the full view [M]); there it was worth +0.0015 ± 0.0004 on the holdout and +0.0003 ± 0.0008 on val at 10 epochs [M]. Next to W16 it reads +0.00052 ± 0.00017 (holdout) / +0.00051 ± 0.00025 (val) nats [M]. It is fresh on train rows and stale on test rows, so it is admissible only in a lag-built form that does not exist yet.
4. **The NATS screen was not blind, the NDCG residual gate was.** On 11 block x witness cells where the value of a block as a trained input is known and resolved, the tree screen reads the block at 2 SE or more in 11 of 11 [M]. In those cells the trained-input NDCG gain is about twice the tree-screen gain in nats (0.8 to 5 times). The round-2 bar of 0.0016 nats therefore asked for a block worth about +0.001 (val) to +0.002 (holdout) as a trained input; family B delivered +0.0003 to +0.0004 nats, i.e. about +0.0006 to +0.0009.
5. **My prior [I]** that an arm built from this inventory gains +0.001 or more on private: about 0.05 (item click rate alone 0.04, with the density pair and the repeat-exposure columns 0.05 to 0.07). This is the T16 figure (0.05) after the calibration, not above it.

### 0.2 Ranked list: blocks that can be made to pass (a) + (b) + (c)

| rank | block (columns) | (a) does the NN read it? | (b) tree setting, val / holdout | linear NATS next to W16 [M]: holdout; val; val in the private mix | (c) arrays for train / holdout / val / test, parity | build cost |
|---|---|---|---|---|---|---|
| 1 | **Item click rate from train feeds**: `tte_it_ctr`, `tte_it_coec`, `tte_it_pos` (NN-ready: `xp_ctr_c`); W25 twins `w25_it_bctr1/2`, `w25_it_once_bctr1`, `w25_it_coec2`, `w25_it_pos2`, `w25_it_npd` | No. No click statistic is among the 104 dense columns [R docs/02 section 4; M `tree_vs_nn_inputs.txt`]. `xp_ctr_c` was a trained input only with the 8 popularity inputs removed (10 run records, 0 without `--xt_exclude` [M `xp_runs.txt`]) | In every tree of record; never left out of a tree. When the POOL lacks it, B0 carries it: +0.00011 ± 0.00016 / +0.00064 ± 0.00016 (10 epochs), +0.00044 ± 0.00017 / +0.00121 ± 0.00016 (2 epochs) [M `cal_posctrl.txt`]. The 26 W25 target statistics (8 of them item-level) on top of a tree that has `tte_it_*`: +0.00011 ± 0.00007 / +0.00010 ± 0.00003 [M `blocks_arms.txt`] | +0.00052 ± 0.00017; +0.00051 ± 0.00025; +0.00019 ± 0.00038. Tuesday (2 d after the cutoff) +0.00082 ± 0.00042, Thursday (4 d) +0.00023 ± 0.00025 [M `cols_read.txt`, `pmix_read.txt`, `val_by_day.txt`] | All four on disk in the FRESH form: `data/feats/{train,val,test}/` 246,745,836 / 1,389,971 / 2,904,806 rows; holdout = rows of train. **Parity not met as built**: a train row reads days < d (lag 1 d), val 2-4 d, test 6-9 d. No lag-built twin exists | Lag-built column with the per-feed cut of the hz2 view (`nn_x/w12/stale/hz2/feats_L/train/hz2_cut_slot.npy`, all 3,151,208 feeds): about 1 CPU-hour of compute, plus a label-free parity gate. An exact causal replay exists for 34% of the train rows only [R reports/STALE_1115/hz2.md section 2] |
| 2 | **Catalogue density** (label-free): `kn_cos1`, `kn_cos20` (cosine of the candidate to its nearest much-shown train items) | No [I]: no NN input relates a candidate to other train items; retrieval and the `ctx_*` columns stay inside the feed and its history [R docs/02 sections 4-5] | Inside B3 (importance rank 10 and 15 of 247 [R famB.md 2.4]). Alone in a reduced-scale tree: +0.00023 ± 0.00025 / +0.00010 ± 0.00010 [R famB_review.md section 4] | +0.00026 ± 0.00012; +0.00030 ± 0.00019 (holdout coefficients); -0.00013 ± 0.00028. Unstable on val: `kn_cos1` alone reads 0.00000 on Tuesday and +0.00083 ± 0.00046 on Thursday; the pair reads +0.00013 ± 0.00016 on M >= 50 [M `blocks_lin.txt`, `bundle_read.txt`, `pmix_read.txt`, `val_by_day.txt`] | `nn_x/w12/cb/famB/feat/c_{xm,hold,val,test}.npy`: 19,741,607 / 4,630,966 / 1,389,971 / 2,904,806 rows. **222,373,263 train rows are missing.** Distribution identical on the four sets [R famB.md 5a]; index membership uses all 15 train days (not causal) | As built, for all train rows: about 22 minutes at 12 threads (104 s per 19.7M rows [R `famB/logs/b3_build_c.log`]). A causal index (items eligible as of day d - lag) is new code: 0.5 to 1 CPU-day with checks [I] |
| 3 | **Same-user earlier impressions**: "this item was shown to this user before" and what the user was shown and did not click (`ua_item_imp/pos`, `tte_ui_imp`, block a of family B) | No: the history holds events, not impressions [R docs/02 4.1] | Only as part of B3: B3 - B0 on the feeds whose user has an earlier train feed +0.00084 ± 0.00030 / +0.00048 ± 0.00019, elsewhere +0.00022 ± 0.00019 / +0.00026 ± 0.00009 [R famB_review.md section 1] | Block a: +0.00003 ± 0.00003; +0.00002 ± 0.00003. `tte_ui_imp` alone: +0.00011 ± 0.00007; +0.00021 ± 0.00012 [M `blocks_lin.txt`, `cols_read.txt`] | Block a: same files and row counts as rank 2 (`a_*.npy`), lag-built twins for xm and holdout only. **Parity fails as built** (source age 72 h on train rows, 240 h on val, 357 h on test; under the test lag 5.4% of train feeds have a source against 29% of test feeds [R famB.md 2.2, 5a]) | Lag-built counts for all train rows: minutes of compute (block a took 23 s for 28.7M rows [R]); the coverage mismatch cannot be built away |
| 4 | **Hard-label tree score as an input** (carrier for 1-3 and for the tree's own +0.00117 / +0.00051 nats) | It is a function of NN inputs plus `tte_*` | B0 next to W16: +0.00117 ± 0.00038 / +0.00051 ± 0.00017; 0.9 / 0.1 blend -0.00006 / -0.00008 [R famB.md] | not a column block | val / holdout / test scores on disk; **no out-of-fold score for any train row** [M `arrays_check.txt`] | 5 user-fold fits of 70-100 min at 12 threads, then scoring 246.7M rows: 8 to 10 CPU-hours [I]; inherits the freshness defect of `tte_*` |

Reading of the list [I]: rank 1 is the only block with a measured value as a trained input, and that value is of popularity type (large on fresh rows, unresolved on val). Rank 2 is positive on the holdout and not stable on val. Rank 3 has no block-level read at all. Rank 4 is a way to carry 1-3, not new information.

### 0.3 NOT ADMISSIBLE (or not a candidate), with the reason

| block | reason |
|---|---|
| `tte_*` and `xp_ctr_c` as built | fresh on train rows (days < d), stale on test (6-9 d): the parity defect of the task's rule. Admissible only lag-built (rank 1) |
| Family B block a as built (39 columns) | fresh / stale (72 h against 240 h and 357 h); the four recency-weighted columns keep 0.06-0.18 of their train spread on val and 0.01-0.03 on test; the medians of `ua_item_dt` and `ua_dt_last` are shifted by 0.8 to 1.9 train ranges [R famB_review.md 3.2] |
| Family B block b (25 transition and user-cluster x item-cluster click rates) | fails (a): the NN learns the same relation from the same labels (category and content of history and candidate are inputs). Fails (b): +0.00004 / +0.00002 next to [W16, B0] [R famB.md 2.3]; +0.00010 ± 0.00007 holdout, 0.00000 val here [M] |
| Family B blocks c and d, label-aware columns (`kn_coec*`, `dup_coec*`, `kn_ex20`, `dup_ex*`) | no signal: +0.00001 to +0.00003 holdout, +0.00003 to +0.00010 val [M `blocks_lin.txt`]; fresh / stale support (twice the expected clicks behind `kn_ex20` on val / test rows [R famB.md 5a]) |
| Family B block d, label-free columns (`dup_n*`, `kn_cos2/5/10`) | add nothing to the pair of block c (all 9 label-free columns +0.00018 / +0.00006 against +0.00041 / +0.00026 for the pair [M]); in no tree arm |
| B1 native cluster / category CTRs, B2 native `item_id` CTR | live inside the CatBoost model; no array can be exported for train rows. B1 is a null (NATS against B0 -0.00021 ± 0.00014 / +0.00002 ± 0.00006), B2 a null with a level loss [R famB.md 3-4]. User-level native keys leak on CPU [R famB.md 3.2] |
| N1 native embedding calcers (LDA, KNN) | inside the model; the CPU KNN trains on a within-feed statistic and applies a cross-feed one [R famSN_review.md 3.4]; null |
| N1x hand-made `hknn*` (6 columns) | NN-derived: mean cross-fitted teacher score of the nearest rows. For a student it is a function of teacher outputs, not information. Arrays for 5,126,377 of the xm rows only, none for test [M `arrays_check.txt`]; null in a tree [R famN.md] |
| N1x hand-made `hlda*` (3 columns) | the label-fitted direction on u * e is the cosine the NN already has (correlation 0.67 with `cos_u` [R famN.md]); `[e, e * u]` is a literal NN input; no test array |
| N2 text token sets, N3 quantisation / one-hot / selection | not arrays; label-free re-encodings of category and cluster counts; null |
| S3 embedding block (128 columns), audit `emb` and `prod` blocks (384 columns) | fail (a): the candidate vector and its product with the user vector are literal inputs of the candidate token [R docs/02 4.1]. S3 halves the NATS of the student [R famS.md]. No train array (audit: val / holdout / test only; S3: rebuilt in RAM) |
| Audit `seq` block (58 per-history-event columns) | fails (a): the NN reads the history tokens; residual R² -0.005 / -0.003 [R audit.md 3.3] |
| W25 G1 and G5 (26 group-level EASE / RDLAE scores) | fail (a) in kind (history co-occurrence, which `seq_cv_*` and the pre2 encoder scores already carry [I]); next to W16 +0.00010 / +0.00001 (G1), +0.00009 ± 0.00010 / +0.00020 ± 0.00010 (G5) [M]; G1 on top of a tree +0.00008 ± 0.00005 / +0.00007 ± 0.00002 nats [M]. As built the weight matrices contain each train user's latest snapshot (my reading of `nn_x/w25/build_lae.py` [M, code read]), which is against the letter of the later-snapshot rule for train rows of that user; a leave-user-fold-out refit would be needed |
| W25 G2, G3, G4 as built (36 contact-ratio columns) | fresh on train rows (days < d); train arrays for 29,734,518 of the 246,745,836 rows. Their item-level part is rank 1 (same information, +0.00068 / +0.00067 [M]); the category / cluster part is nothing next to W16 (+0.00005 / +0.00010 [M]) and learnable from labels; the user part is rank 3 and reads +0.00001 / +0.00003 [M] |
| W25 excluded group (11 columns, PSI above 0.5) | parity-broken by measurement [R `nn_x/w25/logs/psi_w25.log`]. It has the largest read of the inventory on val (+0.00187 ± 0.00047, holdout +0.00074 ± 0.00020 [M]); the columns behind it are the item click rate under another prior (rank 1) and `w25_it_dsf` / `w25_it_dsl` (days since the item was first / last shown: +0.00138 ± 0.00041 on val, +0.00025 ± 0.00011 on the holdout [M `cols_read.txt`]), i.e. the shown-in-train flag: the cold / warm staleness axis that the popularity-light pool already addresses, not a new input |
| Distilled tree students, KNN-on-teacher columns, baseline-on-NN boosting | NN-derived; a student carries a subset of the hard-label tree's information [R famS.md section 4] |
| `tte_u_*`, `tte_uc_*` (5 user-level columns of the 177) | feed-constant or nearly so; read together with `tte_ui_*`: +0.00025 ± 0.00013 / +0.00008 ± 0.00006 [M]; `tte_uc_ctr`, `tte_uc_imp` show an excess lift of +0.022 to +0.028 on feeds with a later same-user feed in the day-1 audit [R reports/leak_audit_features.json], not explained on record |

### 0.4 What "NATS" is, and what a nats gain means for a trained input

NATS(X) = mean over feeds of loss(z_W16) - loss(z_W16, z_X), where the loss is the listwise softmax (conditional logit of the click inside the feed, target y / n_pos) and the one or two coefficients are fitted on the witness itself [R `nn_x/w12/cb/nats.py`]. It is the likelihood information, in nats per feed, that score X adds to the pool. It is never negative by construction. Round 2 called an arm PROMISING at 0.0016 nats on val AND holdout.

Calibration [M, section 1]: for input blocks that stored members were trained with and without, I read (i) the true value as a trained input, (ii) what the same tree screen reads for the block when the pool lacks it, (iii) what a linear composite of the block's columns reads next to the pool that lacks it.

| mapping, on the 11 cells where the trained-input value is positive and resolved (z >= 1.6) | median | range |
|---|---|---|
| tree-screen NATS gain / trained-input NDCG gain | 0.47 | 0.20 to 1.28 |
| linear-composite NATS / trained-input NDCG gain | 0.59 | 0.19 to 1.29 |
| NDCG gain of the pool per nat of the pool's own likelihood gain | 0.26 | 0.13 to 0.92 (record tree row: 0.19 / 0.16) |
| cells where the tree screen reads the block at 2 SE or more | 11 of 11 | |

So: **trained-input NDCG ≈ 2 x tree-screen nats (0.8x to 5x)**. The reverse does not hold: in 4 of 9 cells where the trained-input value is NOT resolved the tree screen still reads +0.00014 to +0.00087 at 2.8 SE or more, and the linear composite reads 2 SE or more in 6 of 9. A positive nats read makes a block eligible; it does not prove a trained-input gain.

### 0.5 Selection context

- CatBoost round 1: 13 tree models and 2 bundles read with the member rule; 6 residual-form block tests and 1 joint fit; 293 stored feature files scanned against the tree residual [R audit.md].
- CatBoost round 2: 51 tree fits read with the screen (family B 7, family N 34, family S 10) and 10 bundles; 81 new columns screened one by one before the arm; 8 pilots; the reviews added 18 reduced-scale fits [R famB.md, famN.md, famS.md, famB_review.md].
- W25: 73 columns, 8 arms, 6 paired contrasts, one residual read per column [R nn_x/w25/out/results.txt].
- "Best NATS +0.00147 / +0.00080" is the largest of 7 family-B arms. Its excess over B0 is +0.00030 / +0.00029. The seed-to-seed sd of NATS is 0.0001-0.00015 / 0.00005-0.00007 [R famB.md section 5], so the best of 7 pure-noise draws around B0 would read about +0.00015 to +0.0002 / +0.00007 to +0.0001 above B0. The B3 arm was declared before its read and replicated at a second seed (+0.00144 / +0.00081), so its gain over B0 is real and small, as the record says.
- My own reads: 27 block composites, 34 single columns, 10 positive controls x 3 trees, on two witnesses. The same pipeline on Gaussian-noise columns [M `noise_null.txt`] reads on val: mean +0.00003 to +0.00006, 95% point +0.00008 to +0.00022, largest of 30-40 draws of 1 or 5 columns +0.00025 to +0.00038; on the holdout: mean +0.00002 to +0.00005, 95% point +0.00005 to +0.00013, largest +0.00009 to +0.00016. On val only the item click-rate family (+0.0005 to +0.0008) is clearly above the best noise draw; the density pair is at it. The linear NDCG change of a noise block has sd 0.00013 on val and 0.00005-0.00008 on the holdout, so no linear NDCG number in this report is evidence.

### 0.6 What follows for the arm [I]

- If one arm is built from this inventory: add to the full view the lag-built item click rate (one or two columns) and, as an option, the label-free pair `kn_cos1` / `kn_cos20` (not stable on val, and its index is not causal). Do not add the user-level label statistics, the transition tables, the W25 scorers or any embedding block.
- CPU pre-gate before any GPU: the NATS of the LAG-BUILT click rate next to W16 on val, with coefficients fitted on a lag-built holdout. The fresh column reads +0.0005; the record says the item click rate keeps under half of its residual at a gap of 8 to 10 days (+0.00051 to +0.00022) and that its AUC falls from 0.5225 at 1 day to 0.501 at 8 days [R reports/EXTERNAL_SCAN_1113.md line 199]. Below +0.0002 on val the column is not worth a student.
- The holdout must not carry the decision: the click rate is a popularity-type input (trained-input value +0.0015 ± 0.0004 on the holdout, +0.0003 ± 0.0008 on val at 10 epochs [M]).
- A zero-build variant exists for the fine-tune stage: all three ranked blocks are on disk for val and test rows, and fine-tune extras are val / test-only inputs by design [R docs/02 4.4]. One early run with `tte_*` as fine-tune extras is on record and was not better (out-of-fold 0.37505 against 0.37534 for the same member without them, one seed, 2026-11-11, an old base [R `nn/logs/nopos_itemtte_m3_ft.log`, `nopos_item_m3_ft.log`]): no evidence either way.

## 1. Calibration: what the screens read for blocks whose trained-input value is known [M]

Script `cal_posctrl.py`, outputs `out/cal_posctrl.txt`, `out/cal_table.txt`, `out/cal_table2.txt`. My NATS code reproduces the shared tool: B0 +0.00117 ± 0.00038 / +0.00051 ± 0.00017 as on record; `nats.py` itself prints +0.00100 / +0.00048 for the CatBoost of record (`out/xcheck_nats_tool.txt`; the 0.00090 / 0.00044 quoted in the round-2 plan is the audit's own code).

Design. For a block B the stored members trained WITHOUT B form the pool "abl"; their twins trained WITH B (same seeds, same recipe) form the pool "twin".
- true = NDCG(twin) - NDCG(abl): the value of B as a trained input at pool level.
- tree = NATS(B0 | abl) - NATS(B0 | twin): the round-2 screen applied to a pool that lacks a block the tree has. B0 has every control block among its 177 features.
- linear = NATS of a composite of B's own columns next to abl; composite coefficients fitted on one user half and read on the other.

| control | epochs, seeds | witness | true trained-input value | tree screen (B0) | linear composite | tree / true | linear / true |
|---|---|---|---|---|---|---|---|
| 8 popularity / co-visitation inputs | 10, 4 | holdout | +0.00852 ± 0.00066 | +0.00443 ± 0.00037 | +0.00834 ± 0.00066 | 0.52 | 0.98 |
| | | val | +0.00160 ± 0.00102 (z 1.6) | +0.00087 ± 0.00024 | +0.00254 ± 0.00055 | not resolved | |
| same | 2, 2 | holdout | +0.00698 ± 0.00066 | +0.00514 ± 0.00048 | +0.00899 ± 0.00069 | 0.74 | 1.29 |
| | | val | +0.00256 ± 0.00101 | +0.00179 ± 0.00033 | +0.00269 ± 0.00057 | 0.70 | 1.05 |
| ONE exposure count `xp_imp_c50` on the popularity-free view | 10, 4 | holdout | +0.00201 ± 0.00046 | +0.00056 ± 0.00016 | +0.00099 ± 0.00023 | 0.28 | 0.49 |
| | | val | +0.00020 ± 0.00077 | -0.00034 ± 0.00018 | +0.00057 ± 0.00026 | not resolved | |
| same | 2, 2 | holdout | +0.00125 ± 0.00042 | +0.00160 ± 0.00018 | +0.00093 ± 0.00022 | 1.28 | 0.74 |
| | | val | +0.00131 ± 0.00071 | +0.00061 ± 0.00018 | +0.00055 ± 0.00026 | 0.47 | 0.42 |
| ONE item click rate `xp_ctr_c` on the popularity-free view | 10, 4 | holdout | +0.00149 ± 0.00044 | +0.00064 ± 0.00016 | +0.00085 ± 0.00022 | 0.43 | 0.57 |
| | | val | +0.00033 ± 0.00075 | +0.00011 ± 0.00016 | +0.00049 ± 0.00025 | not resolved | |
| same | 2, 2 | holdout | +0.00156 ± 0.00040 | +0.00121 ± 0.00016 | +0.00092 ± 0.00023 | 0.78 | 0.59 |
| | | val | +0.00196 ± 0.00069 | +0.00044 ± 0.00017 | +0.00050 ± 0.00025 | 0.22 | 0.26 |
| 5 pre2 encoder scalars | 10, 2 | holdout | +0.00059 ± 0.00049 | +0.00037 ± 0.00012 | +0.00132 ± 0.00026 | not resolved | |
| | | val | +0.00061 ± 0.00083 | +0.00019 ± 0.00025 | +0.00340 ± 0.00064 | not resolved | |
| same | 2, 2 | holdout | -0.00006 ± 0.00036 | +0.00016 ± 0.00005 | +0.00088 ± 0.00021 | not resolved | |
| | | val | +0.00151 ± 0.00065 | +0.00043 ± 0.00013 | +0.00190 ± 0.00048 | 0.29 | 1.26 |
| 10 `ctx_nd` inputs | 2, 2 | holdout | -0.00003 ± 0.00037 | +0.00014 ± 0.00005 | +0.00043 ± 0.00015 | not resolved | |
| | | val | +0.00133 ± 0.00065 | +0.00026 ± 0.00011 | +0.00025 ± 0.00017 | 0.20 | 0.19 |
| `ctx_hn` + `ctx_hs` inputs | 2, 2 | holdout | -0.00044 ± 0.00033 | -0.00003 ± 0.00002 | +0.00013 ± 0.00008 | not resolved | |
| | | val | -0.00015 ± 0.00060 | -0.00021 ± 0.00009 | +0.00044 ± 0.00023 | not resolved | |

Members: abl = `abl_e10_xdrop_pop_s951-954`, `abl_e10_xdrop_pre2_s951-952`, `abl_xdrop_{pop,ctxnd,ctxhn_hs,pre2}_s23-24`; twin = `w5_e10_base` (same seeds), `abl_e10_pop1_s951-954`, `abl_e10_pop2_s951-954`, `abl_base_s23-24`, `abl_pop1_s23-24`, `abl_pop2_s23-24`. The first holdout row reproduces the record's +0.00852 ± 0.00066 [R reports/RECHECK_1114/skeptic_duplicates.md 1.3].

Readings [M unless tagged]:
- **Sensitivity.** In the 11 resolved cells the tree screen is at 2 SE or more every time and the linear composite 10 times. In z units the nats reads are as sharp as the trained-input comparison itself (first row: z 11.8 and 12.6 against 12.9). The record's "10% to 26% sensitivity" is true of the NDCG form of the residual gate (+0.00086 and +0.00222 for the first row [R]); it is not true of the nats form.
- **Size.** Tree-screen nats / trained-input NDCG: median 0.47 (0.20 to 1.28), slope through the origin 0.59 (0.50 without the two popularity holdout cells). Linear: median 0.59 (0.19 to 1.29).
- **Specificity is poor.** In the 9 cells where the trained-input value is not resolved, the tree screen is at 2 SE or more in 4 (the largest, +0.00087, is the val cell of the 8 popularity inputs, where the true value is +0.0016 ± 0.0010) and the linear composite in 6. The pre2 scalars read +0.0034 nats in linear form on val for a trained-input value of +0.0006 ± 0.0008. Information in likelihood is not always ranking gain at the head, and a network that is trained without a block replaces part of it.
- **The member route returns a small part of a block's value.** B0 has the 8 popularity inputs; blended at 0.1 with the pool that lacks them it adds +0.00116 ± 0.00023 on the holdout (+0.00022 next to the pool that has them), against +0.00852 when the inputs are trained in (14%), and -0.00006 ± 0.00039 on val. That is why a block can fail the member rule and still be worth a student arm.
- **What the round-2 numbers mean on this scale [I from M].** The bar (0.0016 on both) needed a block increment of +0.0005 (val) and +0.0011 (holdout) over B0, i.e. a block worth about +0.001 / +0.002 as a trained input. B3 minus B0, +0.00041 ± 0.00016 / +0.00030 ± 0.00008 at two seeds each [R famB.md section 5], maps to about +0.0009 / +0.0006 (range +0.0003 to +0.002) on rows as fresh as the ones it was read on. The W25 columns, +0.00009 ± 0.00005 / +0.00002 ± 0.00002 (section 4), map to +0.0002 / +0.0000.
- **Limits of the calibration.** Hard-label members, not distilled students. 8 of the 11 resolved cells are popularity-type blocks. The tree was not refitted without the block. Pools of 2 to 4 members: the val SE of a true value is 0.0006 to 0.0010, which is why 5 val cells are unresolved.

## 2. Block by block

### 2.1 Definition, kind of information, and whether the neural ranker reads it

NN inputs on record [R docs/02_SOLUTION.md 4.1-4.2, 5]: candidate token = linear map of [unit item vector, its product with the unit user vector] + category embedding + MLP of 104 dense columns; history tokens (item vector, event type, source, category, recency) for the 256 newest events; top-16 retrieval of history items by cosine; cross and set attention. The 104 dense columns are f1-f5 forms, similarity and category-count summaries of the own history, feed context, and 34 stored columns: 3 co-visitation, 5 history-occurrence counts of the item, 20 within-feed context columns, 6 pre2 encoder scores. Of the 177 tree features 33 are among those stored columns; none of the 12 `tte_*` is [M `tree_vs_nn_inputs.txt`].

| # | block | columns | definition | kind of information | does the NN read the same information? |
|---|---|---|---|---|---|
| T | `tte_*` (tree base) | 12 | item / user / user x category / user x item / category impressions, clicks, smoothed click rate, clicks over expected, from train labels [R `src/features_tte.py`] | other users' clicks and exposure; the user's own earlier feeds | NO for clicks. Exposure: the popularity-light members read `xp_imp_c50` = capped `tte_it_imp`; the full view reads history-occurrence counts, not feed impressions |
| Ba | family B block a | 39 | the user's own earlier train impressions: clicks / impressions / clicks over expected by category and cluster, recency-weighted variants, cosine of the candidate to earlier clicked and shown-not-clicked candidates, same item shown / clicked before | what the user was shown and did not click | NO (the history holds events only) |
| Bb | family B block b | 25 | click rate of candidate key c given history key h (category, 256- and 1024-cluster; newest 1 / 5 / 20 events); user-cluster x item-cluster lifts; key base rates | cross-user label tables on content keys | YES in substance: same labels, and both keys are functions of NN inputs |
| Bc | family B block c | 6 | 20 nearest much-shown train items of the candidate inside its 4096-cluster: their clicks over expected (3), expected clicks (1), mean and highest cosine (2, label-free) | neighbours' clicks; catalogue density | clicks: learnable as a function of the item vector. Density: NO |
| Bd | family B block d | 11 | near-duplicate counts above cosine 0.99 / 0.97 / 0.95 / 0.90, cosine of the 2nd / 5th / 10th neighbour (7, label-free); clicks over expected and expected clicks of the near-duplicates (4) | as Bc | as Bc |
| B1 | native CatBoost CTRs on category and 256 / 1024 / 4096 cluster | in-model | ordered target statistics with 255 borders, combinations up to 4 | click rate of content keys | YES in substance |
| B2 | native CatBoost CTR on `item_id` | in-model | ordered item click rate on 436,320 feeds | item click rate (a weaker `tte_it_ctr`) | NO, see T |
| N1 | native embedding calcers on u * e, e - mean history, [u ; e] | in-model | LDA direction on clicks; KNN mean of clicks or of teacher scores | learned similarity metric; neighbour labels | YES ([e, e * u] is an input; learned layers) |
| N1x | hand-made `hlda_*`, `hknn50_*`, `hknn200_*` | 9 | the same, out-of-fold by user; KNN target = cross-fitted teacher score | as N1; NN-derived | YES / it is NN output |
| N2 | text token sets | in-model | candidate category x history categories, candidate cluster x history clusters, bag of words | category / cluster co-occurrence in the own history | YES (history category embeddings, in-model category counts) |
| N3 | quantisation, one-hot, feature selection | none | re-encodings of the 177 | none | n/a |
| S3 | embedding block of family S | 128 (pilots 128 raw, 512 raw) | 32 PCA projections of the candidate vector and their products with the user, mean-history and last-5 vectors | raw content geometry | YES (literal inputs) |
| AUe / AUp / AUs | audit blocks | 128 / 256 / 58 | raw candidate vector; element-wise products with user and mean-history vectors; cosine, type, source of the 12 newest and 5 most similar history events | content geometry; per-event similarity | YES |
| W-G1 | W25 group-level autoencoder scores | 15 | EASE-DAN, RDLAE, EASE on user x group matrices (4096 / 1024 clusters, categories); feed scored from its own history | history co-occurrence of clusters across users (label-free) | YES in kind: `seq_cv_*` (item co-visitation) and the pre2 next-item encoder; NO in this exact form |
| W-G5 | W25 weighted variants of G1 | 11 | type-1 and recency-weighted input vectors | as G1 | as G1 |
| W-G2 | W25 ratio weighting | 6 | cluster contact ratio (Bayesian), type-1 share per cluster, products with the scores | cluster click rate x G1 | YES in substance |
| W-G3 | W25 target statistics | 26 | impressions, positives, Bayesian CTR, clicks over expected, once-per-user counts, distinct positive days for item, category, cluster, user, user x category, user x cluster; all-train and exploration-only sources | as T, with once-per-user and exploration-only variants | as T |
| W-G4 | W25 crosses | 4 | category equals the category of the user's last train positive / of the newest history event; ratio products | last-contact category | YES (history categories with recency) |
| W-Gx | W25 excluded columns | 11 | days since the item's first / last impression, raw count and data-dependent-prior statistics | item age in the data; the same click rates | parity-broken |
| TS | tree scores, leaves | 1 / 64 | score of a hard-label tree; leaf indices reduced by SVD (C01) | a function of the 177 | function of NN inputs + T |

### 2.2 Measured value in the tree setting

Base and unit are stated per row. Level differences below 0.003 between two single tree fits are seed noise [R nn_x/w25/out/results.txt, noise calibration]; the NATS differences are paired per feed.

| block | read (val / holdout) | against | importance / ablation | source |
|---|---|---|---|---|
| Ba + Bb + Bc together (70 columns) | NATS +0.00041 ± 0.00016 / +0.00030 ± 0.00008; level +0.0025 ± 0.0008 / +0.0009 ± 0.0004 (two seeds each). In the private mix the val NATS gain is +0.00031 ± 0.00021; val M >= 50 +0.00031 ± 0.00019 | B0, next to W16 | 8.9 of 100: c 4.65 (`kn_cos1` 2.03, `kn_cos20` 1.76), b 3.64 over 25 columns, a 0.58. No full-scale ablation by block | [R famB.md 2.4, 5]; private mix [M `pmix_read.txt`] |
| the same, split at reduced scale (LightGBM, 30,000 feeds, 3 replicates) | all 70: +0.00018 ± 0.00026 / +0.00030 ± 0.00012; 18 click-free columns +0.00032 ± 0.00026 / +0.00017 ± 0.00011; 52 click statistics +0.00025 ± 0.00026 / +0.00004 ± 0.00009; density pair +0.00023 ± 0.00025 / +0.00010 ± 0.00010; all 70 lag-built +0.00042 ± 0.00028 / +0.00007 ± 0.00010, and +0.00001 ± 0.00011 on a lag-built holdout | the 177 features, next to W16 | | [R famB_review.md section 4] |
| Bd | in no arm | | per-column train-side screen only: label-aware columns dead | [R famB.md 2.3] |
| B1 | NATS -0.00021 ± 0.00014 / +0.00002 ± 0.00006; level -0.00002 ± 0.00078 / +0.00018 ± 0.00046 | B0 | `category_id` 1.94, clusters 0.36 / 0.21 / 0.20 | [R famB.md 3.3] |
| B2 | NATS -0.00005 ± 0.00020 / 0.00000 ± 0.00009; level -0.00203 ± 0.00097 / -0.00302 ± 0.00058 | B0 | `item_id` 0.57 | [R famB.md section 4] |
| N1, N1x, N2, N3 (14 arms) | every NATS difference within ± 0.00014 / ± 0.00007 | same-loss reference at 30,000 or 6,000 feeds | `t_cat` 1.4, `e_pair` 1.25, `hknn50_ehm` 3.0-3.5 | [R famN.md sections 1-3] |
| S3 | NATS +0.00030 / +0.00023 against +0.00062 / +0.00036 without it; level -0.00018 ± 0.00087 / +0.00036 ± 0.00049 at equal rounds | the RMSE student S1f | 12.9% of the model | [R famS.md 2.2] |
| AUe / AUp / AUs | residual form on the tree-reproducible part of W16: NDCG -0.00053 ± 0.00058 / +0.00065 ± 0.00033 (emb), +0.00072 ± 0.00053 / +0.00036 ± 0.00030 (prod), +0.00023 ± 0.00039 / +0.00005 ± 0.00021 (seq); joint fit +0.00115 ± 0.00084 / +0.00100 ± 0.00051; the 0.9 / 0.1 blend with W16 stays at or below zero | LightGBM fitted to z(W16) on 177 features | gain share 0.044 / 0.030 / 0.012 | [R audit.md 3.3-3.4] |
| W25, all 62 used columns | NATS +0.00009 ± 0.00005 / +0.00002 ± 0.00002 (134K feeds), +0.00013 ± 0.00006 / +0.00007 ± 0.00003 (485K feeds). Level +0.00067 ± 0.00045 / -0.00012 ± 0.00025 and +0.00025 ± 0.00044 / +0.00069 ± 0.00025 | the same tree without them, next to W16 | 7.9 of 100; top `w25_c_coec2` 1.79, `w25_rdlae_exp05` 0.65, `w25_grp_t1share` 0.63, `w25_c_once_bctr1` 0.61, `w25_it_once_imp1` 0.58 | NATS [M `blocks_arms.txt`]; level and importance [R nn_x/w25/out/results.txt] |
| W-G1 on top / alone | NATS +0.00008 ± 0.00005 / +0.00007 ± 0.00002 on top; -0.00001 ± 0.00005 / -0.00006 ± 0.00002 alone. Level +0.00085 / +0.00147 on top, -0.00180 / -0.00202 alone | leave-one-group-out, add-one-group | | as above |
| W-G3 on top | NATS +0.00011 ± 0.00007 / +0.00010 ± 0.00003; level +0.00114 ± 0.00047 / +0.00213 ± 0.00028 | leave-one-group-out | | as above |
| seed noise of the same statistics | NATS 0.00000 ± 0.00005 / -0.00008 ± 0.00003; level -0.00086 / -0.00203 | W25 A1 seed 1 minus seed 0 | | as above |
| T (`tte_*`) | never left out of a full-scale tree. `select_features` keeps `tte_it_ctr`, `tte_it_coec`, `tte_it_imp` and drops the other 9; `tte_it_imp` is rank 12 by loss change | | | [R famN.md section 4] |
| TS | B0 +0.00117 ± 0.00038 / +0.00051 ± 0.00017 nats; member rule -0.00006 ± 0.00039 / -0.00008 ± 0.00022. C01 leaf-SVD64 as a residual: -0.00264 ± 0.00064 on val | W16; G3 | NN-input form never run | [R famB.md; reports/COVERAGE_MAP.md line 258] |

### 2.3 Arrays on disk, how train rows were built, parity, leak review

Row counts are from the array headers [M `arrays_check.txt` and the listing in section 9]. Train = 246,745,836 rows in 3,151,208 feeds; holdout = 4,630,966 rows in 30,000 feeds (a subset of train); val 1,389,971; test 2,904,806. xm = 134,118 exploration feeds with M >= 50 (19,741,607 rows) [R infra.md 2.4].

| block | train | holdout | val | test | how a train row was built | matches val / test rows? | leak review on record |
|---|---|---|---|---|---|---|---|
| T, `xp_imp_c50`, `xp_ctr_c` | `data/feats/train/<name>.npy`, 246,745,836 | inside train | `data/feats/val`, 1,389,971 | `data/feats/test`, 2,904,806 | train days before the row's day (expanding window by day); own feed excluded by day; not leave-user-out; the 30,000 holdout feeds are sources | NO: lag 1 d on train rows, 2-4 d on val, 6-9 d on test; val / test read all 15 days. PSI 0.014 / 0.10 for `xp_ctr_c` [R ledger E_kd_predeclared.txt line 587] | day-1 audit: no later-feed excess for the item columns, +0.022 to +0.028 for `tte_uc_*` [R reports/leak_audit_features.json]; builder has no holdout handling [R famB_review.md issue 12] |
| lag-built `xp_imp_hz_c50` (hz2 view; exposure count only) | `nn_x/w12/stale/hz2/feats_L/train/`, 246,745,836; per-feed lag and cut for 3,151,208 feeds | inside train | stored val / test columns are the lag-built form by nature | | per-feed test-like lag of 6-9 d, strictly causal | yes on days from Sep 13; Sep 6-8 (27% of train feeds) cannot be made test-like [R hz2.md] | reviewed [R reports/STALE_1115/hz2_review.md] |
| Ba, Bb, Bc, Bd | `nn_x/w12/cb/famB/feat/{a,b,c,d}_xm.npy`, 19,741,607 x 39 / 25 / 6 / 11. NOT for the other 222,373,263 train rows | `{a,b,c,d}_hold.npy`, 4,630,966 | `_val.npy`, 1,389,971 | `_test.npy`, 2,904,806 | a: the user's non-holdout train feeds with a strictly earlier timestamp. b, c, d click columns: the 4 other user folds, earlier days only, holdout never a source. c, d label-free columns: index = items with 5 or more impressions over ALL 15 train days | a: NO (age and support). b, c, d click columns: same ratio distributions, different support (NaN 8.6% on xm, 0% on val / test) and age. `kn_cos1`, `kn_cos20`: identical p10 / p50 / p90 on the four sets; membership of the index is not causal [R famB.md 5a, section 7] | famB_review: no leak; 19 columns rebuilt from raw data on 5,000 train and 4,996 val rows (counts exact, cosines to 9e-8) |
| Ba, Bb, Bc lag-built | `feat/lag_{a,b,c}_xm.npy` | `lag_{a,b,c}_hold.npy` | not needed | not needed | a train row on day d reads days up to d - L, L drawn from 6-9 per feed | closer in age, further in support: 5.4% of xm feeds have a user source, 41% a block b / c statistic [R famB.md 5a] | lag build of block a checked on 3,000 feeds [R famB_review.md 3.1] |
| N1x | `nn_x/w12/cb/famN/cache/hx/n30k/*_xm.npy`: 5,126,377 finite rows of 19,741,607 (the 30,000 + 5,000 feed subset) | `*_hold.npy` | `*_val.npy` | NONE | other user folds of the fit subset | same distribution and label association on fit / selection / holdout / val [R famSN_review.md section 2] | 0.33% of holdout feeds have their user in the fit subset with a LATER feed [R same] |
| S3 | none (`famS/cache/pca32.npz` basis only; block rebuilt in RAM in 15 s) | none | none | none | label-free | standardised mean difference at most 0.077 [R famS.md section 5] | famSN_review: clean |
| AUe / AUp / AUs | NONE | `nn_x/w12/cb/audit/cache/blk_H_*.npy` | `blk_V_*.npy` | `blk_T_*.npy` | label-free | by construction | none needed |
| W25 (73 columns) | `nn_x/w25/data/train/w25_*.npy`, 29,734,518 rows = the feeds with p_random > 0.5 (xm + short exploration-like feeds + holdout), index `rows.npy`. NOT for the other 217,011,318 rows | inside those train rows | `nn_x/w25/data/val/`, 1,389,971 | `nn_x/w25/data/test/`, 2,904,806 | G2-G4: train days before the row's day, day 0 NaN, user 0 NaN; holdout feeds are sources. G1 / G5: weight matrices from ONE train snapshot per user (the latest), score from the feed's own history | G2-G4: fresh / stale as T. PSI train to val below 0.5 for the 62 used columns (G1 / G5 at most 0.005; category statistics 0.15-0.23); 11 excluded [R `nn_x/w25/logs/psi_w25.log`, `psi_w25_v2.log`] | spot check: target statistics equal a strict days-before-d recomputation on 2,500 train and 2,000 val rows; scores equal the own-history recomputation [R `nn_x/w25/review/leak_spotcheck.log`]. Not reviewed: the weight matrices are not leave-user-out |
| B1, B2, N1, N2 | none (statistics inside `.cbm` files) | | | | CatBoost ordered statistics | train / apply mismatch for the native KNN | noise-label tests [R famB.md 3.2] |
| TS | NONE (no out-of-fold score) | `nn_x/w12/cb/infra/out/ts_lr020_s0_hold.npy`, `famB/preds/b3_abc_s{0,1}_hold.npy` | `*_val.npy` | `*_test.npy` | trees fitted on 120,591 xm feeds | | famB_review, famSN_review |

## 3. One protocol for every block: linear composite next to W16 [M]

Script `blocks_read.py blocks`, outputs `out/blocks_lin.txt`, `out/blocks_lin.json`; single columns `cols_read.py` -> `out/cols_read.txt`; unions `bundle_read.py`; private mix `pmix_read.py`; val by day `val_by_day.py`; null `noise_null.py`.
Protocol: per-feed z of each column (NaN to 0) plus a missing flag where the NaN pattern varies inside feeds; coefficients fitted on one user half and read on the other, inside each witness; val also with the coefficients fitted on all 30,000 holdout feeds (train-period labels only). NATS = gain of [z_W16, composite] over [z_W16]. This is the residual form; section 1 says how to read it. Feed-constant columns drop out, so the counts differ from the block sizes.

| block | columns used | holdout | val | val, holdout coefficients | next to [W16, B0]: holdout / val |
|---|---|---|---|---|---|
| T: all 12 `tte_*` | 9 | +0.00050 ± 0.00018 | +0.00078 ± 0.00028 | +0.00071 ± 0.00028 | +0.00041 / +0.00058 |
| T: item and category level | 5 | +0.00059 ± 0.00018 | +0.00055 ± 0.00025 | +0.00066 ± 0.00028 | +0.00047 / +0.00035 |
| T: user level | 4 | +0.00008 ± 0.00006 | +0.00025 ± 0.00013 | +0.00014 ± 0.00010 | +0.00011 / +0.00023 |
| `xp_ctr_c` | 1 | +0.00052 ± 0.00017 | +0.00051 ± 0.00025 | +0.00050 ± 0.00024 | +0.00044 / +0.00039 |
| `xp_imp_c50` | 1 | +0.00009 ± 0.00007 | +0.00009 ± 0.00010 | +0.00009 ± 0.00010 | +0.00009 / +0.00011 |
| Ba, all | 36 | +0.00003 ± 0.00003 | +0.00002 ± 0.00003 | +0.00013 ± 0.00012 | +0.00003 / +0.00001 |
| Ba, lag-built | 36 | +0.00006 ± 0.00006 | | | +0.00006 |
| Bb | 27 | +0.00010 ± 0.00007 | +0.00000 ± 0.00001 | +0.00020 ± 0.00011 | +0.00005 / +0.00003 |
| Bb, lag-built | 27 | +0.00009 ± 0.00007 | | | +0.00008 |
| Bc density pair | 2 | +0.00026 ± 0.00012 | +0.00041 ± 0.00022 | +0.00030 ± 0.00019 | +0.00034 / +0.00038 |
| Bc click columns | 4 | +0.00001 ± 0.00004 | +0.00003 ± 0.00006 | +0.00008 ± 0.00010 | +0.00002 / +0.00007 |
| Bc click columns, lag-built | 4 | +0.00003 ± 0.00004 | | | +0.00002 |
| Bc `kn_ex20` | 2 | +0.00002 ± 0.00003 | +0.00007 ± 0.00009 | +0.00006 ± 0.00008 | |
| Bd label-free | 7 | +0.00005 ± 0.00005 | +0.00007 ± 0.00009 | +0.00015 ± 0.00013 | +0.00009 / +0.00010 |
| Bd click columns | 5 | +0.00003 ± 0.00006 | +0.00010 ± 0.00011 | +0.00002 ± 0.00005 | |
| W-G1 | 15 | +0.00001 ± 0.00002 | +0.00010 ± 0.00011 | +0.00014 ± 0.00009 | +0.00001 / +0.00001 |
| W-G5 | 11 | +0.00020 ± 0.00010 | +0.00009 ± 0.00010 | +0.00015 ± 0.00013 | +0.00011 / +0.00000 |
| W-G2 | 7 | +0.00006 ± 0.00005 | +0.00046 ± 0.00023 | +0.00046 ± 0.00023 | +0.00002 / +0.00018 |
| W-G3, all | 21 | +0.00069 ± 0.00019 | +0.00060 ± 0.00026 | +0.00087 ± 0.00032 | +0.00062 / +0.00045 |
| W-G3, item level | 6 | +0.00067 ± 0.00019 | +0.00068 ± 0.00028 | +0.00076 ± 0.00030 | +0.00060 / +0.00061 |
| W-G3, category / cluster level | 9 | +0.00010 ± 0.00007 | +0.00005 ± 0.00008 | +0.00025 ± 0.00017 | |
| W-G3, user level | 6 | +0.00003 ± 0.00004 | +0.00001 ± 0.00003 | +0.00001 ± 0.00003 | |
| W-G4 | 4 | +0.00006 ± 0.00006 | +0.00000 ± 0.00001 | +0.00021 ± 0.00016 | |
| W-Gx (parity-broken; information) | 10 | +0.00074 ± 0.00020 | +0.00187 ± 0.00047 | +0.00140 ± 0.00040 | +0.00066 / +0.00185 |
| union: `xp_ctr_c` + density pair | 3 | +0.00078 ± 0.00020 | +0.00068 ± 0.00028 | +0.00079 ± 0.00031 | |

Noise null of the same pipeline (section 0.5): val up to +0.00025 to +0.00038 for the best of 30-40 noise blocks, holdout up to +0.00009 to +0.00016.

Single columns [M `cols_read.txt`], holdout; val with holdout coefficients; transfer beta on val (1 = the holdout coefficient holds on val):

| column | holdout | val | beta |
|---|---|---|---|
| `tte_it_coec` | +0.00061 ± 0.00018 | +0.00047 ± 0.00024 | +0.87 |
| `tte_it_ctr` | +0.00054 ± 0.00017 | +0.00044 ± 0.00023 | +0.90 |
| `xp_ctr_c` | +0.00052 ± 0.00017 | +0.00050 ± 0.00024 | +0.96 |
| `w25_it_bctr2` (exploration-only source) | +0.00049 ± 0.00016 | +0.00039 ± 0.00021 | +0.91 |
| `w25_it_once_bctr1` (once per user) | +0.00048 ± 0.00016 | +0.00045 ± 0.00023 | +0.96 |
| `w25_it_coec2` | +0.00038 ± 0.00015 | +0.00032 ± 0.00019 | +0.93 |
| `tte_it_pos` | +0.00021 ± 0.00011 | +0.00027 ± 0.00018 | +1.15 |
| `tte_it_imp`, `xp_imp_c50`, `w25_it_imp2`, `w25_it_once_imp1`, `it_hist_all_luo` (exposure counts) | +0.00001 to +0.00009 | +0.00001 to +0.00009 | unstable |
| `w25_it_dsf` (days since first shown; parity-broken) | +0.00025 ± 0.00011 | +0.00086 ± 0.00032 | +1.76 |
| `kn_cos1` | +0.00028 ± 0.00012 | +0.00020 ± 0.00015 | +0.84 |
| `kn_cos20` | +0.00014 ± 0.00009 | +0.00002 ± 0.00004 | +0.33 |
| `tte_ui_imp` | +0.00011 ± 0.00007 | +0.00021 ± 0.00012 | +1.97 |
| `tte_c_ctr` | +0.00008 ± 0.00007 | +0.00032 ± 0.00020 | +1.86 |

`tte_ui_pos` failed numerically on the holdout (a near-empty column; the fit diverged) and is not quoted.

Readings [M]:
- **What W16 lacks, in this lens, is the item click RATE, not exposure.** Every form of it (smoothed rate, clicks over expected, once per user, exploration-only source) reads +0.0004 to +0.0006 on both witnesses with the holdout coefficient holding on val (beta 0.87 to 0.96). Exposure counts read nothing: the full view already has them.
- The item-level columns of W25 are the same information as `tte_it_*` (block read, holdout and val: +0.00067 and +0.00068 against +0.00059 and +0.00055); the once-per-user and exploration-only variants are not better than the plain rate.
- Click rate and density are additive: +0.00078 ± 0.00020 on the holdout and +0.00079 ± 0.00031 on val (holdout coefficients) for the three columns.
- **Private mix** [M `pmix_read.txt`; weights of `src/pmix.py`, gains from the unweighted fit]: `xp_ctr_c` +0.00019 ± 0.00038, density pair -0.00013 ± 0.00028, the three columns +0.00006 ± 0.00047 (public mix +0.00030 ± 0.00039). No veto (none is below -1 SE), no support either.
- **Val by day** [M `val_by_day.txt`]: `xp_ctr_c` +0.00082 ± 0.00042 on Tuesday (2 d after the cutoff) and +0.00023 ± 0.00025 on Thursday (4 d): the direction of a decay with the lag, not resolved. `kn_cos1`: 0.00000 on Tuesday, +0.00083 ± 0.00046 on Thursday, +0.00006 ± 0.00011 on M >= 50: no stable val signal.
- Everything else of family B and of W25 is at the noise null of this lens. For blocks a and b this lens is weak (sparse, mostly-missing columns in a linear form); their tree-setting read is the B3 row of 2.2.
- The linear NDCG change is not quoted for any block: its null sd is 0.00013 on val.

## 4. The W25 arms read with the round-2 screen [M]

W25 was run on 2026-11-13, before the NATS screen existed; its record holds level differences only. Script `blocks_read.py arms`, output `out/blocks_arms.txt`. Predictions `nn_x/w25/out/<arm>_{val,trH}.npy`.

| arm | level | NATS next to W16 | val M >= 50 |
|---|---|---|---|
| A0 = CatBoost of record, 177 features, 134K feeds | 0.38201 / 0.30424 | +0.00100 ± 0.00035 / +0.00048 ± 0.00016 | +0.00086 ± 0.00040 |
| A2c = A0 + 62 W25 columns | 0.38268 / 0.30412 | +0.00108 ± 0.00037 / +0.00050 ± 0.00016 | +0.00088 ± 0.00041 |
| A1 = 177 features, 485K feeds | 0.38077 / 0.30237 | +0.00110 ± 0.00037 / +0.00048 ± 0.00016 | +0.00091 ± 0.00042 |
| A2 = A1 + 62 W25 columns | 0.38101 / 0.30306 | +0.00123 ± 0.00039 / +0.00055 ± 0.00017 | +0.00106 ± 0.00045 |

Paired NATS differences are in the W25 rows of 2.2. The 62 columns add at most +0.00013 ± 0.00006 / +0.00007 ± 0.00003, a third of what family B added, and the seed pair of the same recipe differs by -0.00008 ± 0.00003 on the holdout. On the scale of section 1 that is a trained-input value of +0.0002 or less. The W25 closure stands in the nats lens as well.

## 5. Build cost and what parity demands, per ranked block

| rank | what must be built for all four splits | parity form | cost | what limits it |
|---|---|---|---|---|
| 1 item click rate | one or two columns (`tte_it_ctr`-like, `tte_it_coec`-like) for the 246.7M train rows with the per-feed cut of the hz2 view; val and test columns exist | a train row reads train days up to (its day - lag), lag 6-9 d per feed as in `hz2_lag.npy`; missing where the item has no impression inside the window, with a flag | one pass over the train rows: about 1 CPU-hour, then a label-free gate (zero share and quantiles by listing age, train against val and test) [I; the W25 builder wrote 42 such columns in 333 s, R `nn_x/w25/logs/build_te.log`] | exact replay for 34% of the train rows; Sep 6-8 (27% of train feeds) have no past; the lag-parity view of the popularity inputs passed at pool level and was closed at file level (-0.00047 ± 0.00035 on val [R reports/CURRENT_STATUS.md log 2026-11-15T13:10Z]) |
| 2 density pair | 2 columns for the 222.4M missing train rows | as built: index of much-shown items over all 15 train days (label-free). Strict form: index as of the row's cut | as built 22 min at 12 threads; strict form 0.5-1 CPU-day [I] | the strict form has no index for the early train days; the as-built form lets a train row see items that became much-shown later |
| 3 repeat exposure | `ua_item_imp`, `ua_item_pos` (and the category / cluster counts of block a without the recency-weighted and time-since columns) for all train rows, lag-built | the user's train feeds up to (day - lag) | minutes [R block a: 23 s for 28.7M rows] | 5.4% of train feeds covered against 29% of test feeds; sources 10-15 days old on test |
| 4 tree score | 5 user-fold YetiRank fits of the B3 recipe, out-of-fold score for xm feeds, fold-model score for all other train rows, val, test | the tree's inputs must first be lag-built themselves | 8-10 CPU-hours [I] | carrier only |

The student arm itself (GPU, teacher that has not seen the input) is outside this inventory; reports/TRICKS_1115/int_weak.md row T16 prices it at 27 GPU-hours for 8 students.

## 6. Limits and what was NOT checked

- No model was trained. No tree was refitted with or without a block. The tree-setting reads of single blocks are the records' (family B at full scale only as a bundle of three blocks).
- The lag-built form of the rank-1 column does not exist, so its value at the test lag is not measured here. The decay figures are the record's (AUC 0.5225 to 0.501; residual +0.00051 to +0.00022) and the unresolved Tuesday / Thursday split.
- The calibration uses hard-label members at 2 and 10 epochs and pools of 2 to 4 members; whether a distilled student, whose teacher has not seen a new input, gains the same is not measured anywhere.
- Composite coefficients on val come from user-grouped halves of val or from the holdout; the two-parameter mix is fitted on the feeds read (the definition of the screen). Val labels were used for evaluation reads only; nothing was selected for further use.
- The private-mix numbers use the unweighted fit and reweight the per-feed gains; a weighted fit was not run.
- "Does the NN read it" is judged from docs/02_SOLUTION.md sections 4-5 and the name lists of `nn/train.py`; `nn/model.py` was not read line by line.
- The remark that the W25 weight matrices contain each user's latest train snapshot is my reading of the builder's docstring; its size as a leak was not measured (one user in about 2.5 million rows of a group-level matrix).
- NaN shares and distributions of the stored arrays were not recomputed; they are quoted from the records. Row counts are from array headers.
- The 177 tree features that are not NN inputs and are not label statistics (`clu_*`, `cat_*`, extra `seq_*` and `cos_*` summaries) belong to the other inventories and were not read here.
- One numerical failure: `tte_ui_pos` on the holdout.

## 7. Commands and files

Every job: `cd <repo>/reports/INPUTS_1116/catboost_blocks && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=$PWD/cache/numba && nice -n 5 <repo>/.venv/bin/python -u <script> [> out/log_<name>.txt]`. Longest job 21 minutes; never more than two 12-thread jobs at a time (light 2-thread snippets ran next to them); no file was written outside this directory (checked with `find -newermt` on the imported source directories). Imported read-only: `nn_x/w12/recheck/pool_composition/pc_common.py`, `src/pmix.py`, `src/metrics.py`, `src/stack_val.py`.

| script | what | output under `out/` |
|---|---|---|
| `cbi_common.py` | witnesses, per-feed z, vectorised conditional logit, NATS | |
| `cal_posctrl.py` | section 1 | `cal_posctrl.txt`, `.json`, `cal_table.txt`, `cal_table2.txt` |
| `blocks_read.py arms` | section 4 | `blocks_arms.txt`, `.json` |
| `blocks_read.py blocks` | section 3, block table | `blocks_lin.txt`, `.json` |
| `cols_read.py` | section 3, single columns | `cols_read.txt`, `.json` |
| `bundle_read.py`, `pmix_read.py`, `val_by_day.py` | unions, private mix, val by day | `bundle_read.txt`, `pmix_read.txt`, `val_by_day.txt` |
| `noise_null.py` | null of the block read | `noise_null.txt`, `.json` |
| inline snippets | tool cross-check, array headers, tree features against NN inputs, run records that name a column | `xcheck_nats_tool.txt`, `arrays_check.txt`, `tree_vs_nn_inputs.txt`, `xp_runs.txt` |

Records read: `reports/CATBOOST_1115/` (all ten files), `reports/COVERAGE_MAP.md` (lines 111, 149, 217, 234, 258, 323, 327, 328, 341, 347), `nn_x/w25/` (plan, results, builders' headers, PSI and cell logs, review logs), `nn_x/w12/cb/nats.py`, `nn_x/w12/cb/PREDECLARED.txt`, `docs/02_SOLUTION.md` sections 1-6, `docs/05_METHODOLOGY.md` section 5, `reports/CURRENT_STATUS.md` section 3 and the logs of 2026-11-15, `reports/RECHECK_1114/skeptic_duplicates.md` 1.3, `reports/TRICKS_1115/int_weak.md` rows T09, T15, T16, `reports/STALE_1115/hz2.md` sections 0-2, `reports/EXTERNAL_SCAN_1113.md` line 199, `nn_x/abl/logs/abl_eval_pop1.txt`, `abl_eval_pop2.txt`.
