# hz2n: no-ID encoder score on the rows a training view makes cold (builder report)

Builder run 2026-11-15 07:12:44Z to 07:47Z, one run, CPU only. Work directory `nn_x/w12/stale/hz2n/`. Hand-over file `READY.txt` first written 07:39:44Z (27 minutes after the start), refreshed 07:44:37Z. Marks: [M] measured by me, [D] read in a project file, [I] inferred. ± is not used: nothing in this report is a labelled number. Every criterion was written with its `date -u` stamp in `DECLARATION.txt` (three addenda) before the number it judges.

## 0. Verdict

1. **Built as asked and proven.** `cd_arms6c.py` = `cd_arms6b.py` + an optional column map for the encoder-score inputs; with no encoder file in `HZCD_DIR` it is `cd_arms6b` bit for bit (AST, tensors, 40 batches on 2,000 feeds, log line apart from the tag). The three views exist (`feats_n`, `feats_Ln`, `feats_hn`). On the val rows my row-wise function reproduces the witness files `pre2both_{pre2_fe,pre2_fe_z,pre2_fe_s}.npy` with max|d| = 0 (0 of 1,389,971 rows differ bitwise).
2. **The defect is removed for the fe head.** P(fe-head ID contribution != 0 | view `it_hist_all` = 0): 0.236 / 0.245 / 0.110 before (views a / b / c, all days; 0.302 / 0.317 / 0.135 on days >= Sep 13), 0 rows after; 0 on val / public / private as built [M].
3. **The fix as specified repairs half of the tell.** The second encoder head (`pre2_new_z`, `pre2_new_s`, both model inputs under `cvnd_pre2s`) carries the item-ID contribution on exactly the same rows, and the witness scenario `both_pre2both` does not replace it either. After the as-asked change those rows read "fe head without ID + new head with ID", which exists on no val / test row (0 of 4.29M) and on no as-built train row [M]. Declared extension **F** (`feats_nF`, `feats_LnF`, `feats_hnF`) rebuilds the new head the same way and equals the project's own `both_pre2full` files bitwise on val.
4. **View c (arm hz as trained) has a second train-only combination:** `it_hist_all` = 0 with `seq_cv_pop` > 0 on 9.7% of its cold rows, 0.0000 on val / test [M]. Declared extension `feats_hnFS` = `feats_hnF` + `seq_cv_*` = 0 on view-cold rows.
5. **Train calibration caveat of the as-asked score:** on train rows without any ID contribution the stored `pre2_fe_noid` sits 0.257 / 0.184 / 0.008 below `pre2_fe` (encoder windows Sep 9-11 / 12-15 / 17-20; val +0.003, test 0). The as-asked change therefore also lowers `pre2_fe_s` of made-cold rows that never had an ID, by up to 0.06 private-row SD. Declared sensitivity family **P** removes it; the level evidence does not separate P from as-asked (<= 0.03 SD), so P is not recommended for GPU.
6. **Gate against my declared rules:** G1 PASS (all views). G2(i) PASS on the population exchangeable with test feeds (+0.207 -> -0.003, the review's +0.210 reproduced) and FAIL by the letter on all train feeds (+0.641 -> +0.366), because untouched cold rows of production-ordered feeds already read +0.265 there: my criterion ignored that offset. Reported as declared; not a defect of the view in my reading.
7. **Not done:** `nn_x/w5/train.py` was not run on CPU (the full train split needs more than my 40 GB limit). The loader was instead run over all 246,745,836 rows with the real `setup_hzcd` for all ten directories (0 differing cells) and a 30-step cooldown smoke ran on a 600-feed mini split. The first GPU run is the end-to-end test.
8. **State at 07:43Z (seen read-only):** the lead copied the module to `nn_x/w5/cd_arms6c.py` (md5 equal to mine, `dbe5a1387569ecbcb832de6e4cc97297`), prepared `train.py.6c` and the lane arms, and made `both_pre2full` the deciding stale-val scenario on the basis of finding 3; I verified that finding again afterwards (section 5.1).

## 1. What was built

| HZCD_DIR (under `nn_x/w12/stale/hz2n/`) | base view | HZ2_KEEP | swapped columns (W8 preset) | encoder columns | hz tensor |
|---|---|---|---|---|---|
| `feats_n` (a, as asked) | `hz2/feats` | `hz2/feats/train/hz2_keep_rows.npy` | 10 | `pre2_fe_z`, `pre2_fe_s` | 4.60 GB |
| `feats_Ln` (b, as asked) | `hz2/feats_L` | `hz2/feats_L/train/hz2_keep_rows.npy` | 10 | same | 4.60 GB |
| `feats_hn` (c, as asked) | `data/feats_new` (5 `it_hist_hz_*`) | none | 7 | same | 3.22 GB |
| `feats_nF`, `feats_LnF`, `feats_hnF` (F) | as a / b / c | as a / b / c | 12 / 12 / 9 | + `pre2_new_z`, `pre2_new_s` | 5.52 / 5.52 / 4.14 GB |
| `feats_hnFS` | c | none | 12 | 4, + `seq_cv_invd20/pop/norm` zeroed on view-cold rows | 5.52 GB |
| `feats_nP`, `feats_LnP`, `feats_hnP` (P) | as a / b / c | as a / b / c | 10 / 10 / 7 | fe head on the fe calibration | 4.60 / 4.60 / 3.22 GB |

- Each directory holds symlinks to every base-view file `cd_arms6b` reads plus the new real files. New files per as-asked directory: `train/pre2_hz_fe.npy` (raw mixed score), `train/pre2_hz_fe_s.npy`, `train/pre2_hz_fe_z.npy`, `train/hz2n_newly_cold.npy` (the mask), `aux/pre2_hz_fe_has.npy`. F adds `train/pre2_hz_new.npy`, `pre2_hz_new_s.npy`, `pre2_hz_new_z.npy`; its fe-head files are symlinks to the as-asked ones.
- Module: `cd_arms6c.py`, generated from `nn_x/w5/cd_arms6b.py` by `make_6c.py` (five exact single-occurrence replacements; `cd_arms6c_vs_cd_arms6b.diff`). Same public names. Change in `train.py`: line 56, `from cd_arms6b import ...` -> `from cd_arms6c import ...` (`train_py_import_6c.txt`); the same line is `nn_x/w12/kd/fd/train_full2.py` l.69. `nn_x/abl/train.py` imports `cd_arms6` and is not concerned.
- Column map added: `pre2_fe -> pre2_hz_fe`, `pre2_fe_z -> pre2_hz_fe_z`, `pre2_fe_s -> pre2_hz_fe_s`, `has:pre2_fe -> pre2_hz_fe_has`, `pre2_new -> pre2_hz_new`, `pre2_new_z -> pre2_hz_new_z`, `pre2_new_s -> pre2_hz_new_s`, each swapped only if its file exists in `<HZCD_DIR>/train/`. The view names take the same `attach_xt` branch as the stored names (tested on a probe vector). A head with only part of its files is refused; preset `cvnd_pre2` (which reads `pre2_id_contrib`) is refused with an encoder view.
- Log line of `setup_hzcd`: `[6c: <n> swapped columns [names]; HZ2_KEEP rows on stored values: <k>; ENCODER-SCORE VIEW ON: <m> of them are encoder-score columns [names] <- [files] (no-ID score on rows made cold by the view)]`, followed as before by one `col i name -> file: zero rate, rows changed` entry per column. Expected values per directory: `READY.txt` section 5b.
- Two readers for the read-out, both wrappers that import the existing script unchanged: `stale_pred_full.py` (GPU, scenario `both_pre2full`, not run by me) and `hold_infer_n.py` (CPU scoring of the 30,000 holdout feeds through a new view; its column attach self-test passed, scoring not run).

## 2. Definition, row by row

- View counter, RAW (float32 file, not the log1p / fp16 model scale): a `nn_x/w12/stale/hz2/feats/train/it_hist_hz_all.npy`, b `.../feats_L/train/it_hist_hz_all.npy`, c `data/feats_new/train/it_hist_hz_all.npy`. Stored counter: `nn/feat_cache/train/it_hist_all_luo.npy`. Effective counter as trained: keep ? stored : view. Raw and model scale agree on "cold": 0 rows with view > 0 whose fp16(log1p) is 0; no NaN; 0 rows with view > 0 and stored = 0 [M].
- Newly cold row: nc = (view == 0) & (stored > 0) & ~keep. Rows: a 68,544,931 (27.8% of train rows; 51,286,501 with an encoder score), b 72,310,232 (55,051,802), c 24,039,264 (16,920,295) [M]. The other nc rows are in feeds of Sep 6-8, which lie before the first encoder window and have no score (`has:pre2_fe` = 0 before and after).
- On nc rows (as asked): `pre2_fe` := `data/feats/train/pre2_fe_noid.npy`; `pre2_fe_s` := float32((`pre2_fe_noid` + 20) / 6); `pre2_fe_z` := mean over the two configs (v3, v4) of the within-feed z (`src/stack_val.py::feed_rank_z`, copied verbatim) of the per-config mixed scores, with the alignment of `pretrain/assemble_train.py` (`--wtag "w{k}r_{c}" --rtag "{c}t2"`). This is `build_pre2both` with the train alignment in place of the val alignment.
- Other rows: `pre2_fe`, `pre2_fe_s`, `has:pre2_fe` keep the stored bits on every non-nc row. `pre2_fe_z` keeps the stored bits in every feed without an nc row; in a feed with one it changes on ALL rows, because the witness re-standardises the feed. This is the one place where "every other row keeps the stored value bit for bit" cannot hold together with witness parity; I gave parity priority and declared it (D0 note). Mean |dz| on the non-nc rows of such feeds: 0.059 (a), 0.062 (b), 0.015 (c); the witness on val: 0.0058 [M].
- `has:pre2_fe`: 0 of 246,745,836 rows change in every directory [M] (score and no-ID score have the same NaN mask). Nothing needs swapping; the file is kept in `aux/` so that a provably identical column does not cost 0.46 GB of GPU memory per run. The module swaps it if the file is put in `train/` (tested).
- Keep mask: the two keep files are byte-identical; 0 of 46,918,534 keep rows have a finite `pre2_fe` [M]. Keep rows are never nc and have no score, so the keep semantics of `cd_arms6b` (keep rows read stored values in every swapped column) hold trivially for the new columns and are tested (P4a, P4b).

## 3. Proofs

| | what | result |
|---|---|---|
| P0 | per-config arithmetic without override vs stored `pre2_fe`, `pre2_fe_noid`, `pre2_fe_z` on all 177,782,868 window rows | 0 rows differ bitwise in each window; stored `pre2_fe_s` == float32((`pre2_fe` + 20) / 6) bitwise; 0 rows outside the windows have a score. New head: same, 0 rows [M] |
| P1 | `cd_arms6c` vs `nn_x/w5/cd_arms6b.py`, no encoder file in the view (unset; `hz2/feats` with and without keep; `hz2/feats_L` with keep) | changed defs exactly `luo_cols`, `setup_hzcd`; no new def; constants `ENC_TO_HZ`, `ENC_GROUPS` added; same columns for the W8 preset and the popularity-light list; `load_hz` tensors bit-identical; 40 batches x 22 keys bit-identical on 2,000 random non-holdout train feeds (158,751 cells); `hz_stats` identical; log line identical apart from "6c" [M] |
| P2 | row-wise function on VAL rows with the witness counters (`nn_x/r6/stale_cf/both_it_hist_all_luo.npy`) vs `solvers/out/stale/cf/pre2both_*.npy` | max\|d\| = 0e+00 / 0e+00 / 0e+00 for `pre2_fe` / `pre2_fe_z` / `pre2_fe_s`; 0 rows differ bitwise; nc == `newly_cold.npy`; has identical [M] |
| P2n | new head vs `pre2full_{pre2_new,pre2_new_z,pre2_new_s}.npy` | max\|d\| = 0 / 0 / 0; 0 rows differ bitwise [M] |
| P3 | unchanged rows, all 246,745,836 rows, each directory | rows outside nc with changed bits in score / `_s` / has: 0 / 0 / 0; `_z` rows changed in feeds without an nc row: 0; same for the new-head files [M] |
| P4a | REAL `setup_hzcd` (unwrapped `load_hz`, real keep file) over the full row space, CPU stand-in split with the stored fp16 columns | all 10 directories: 0 cells differ from keep ? stored : fp16(transform(file)) in every swapped column; encoder `_s` columns: 0 rows changed outside nc, 0 nc rows that are not the no-ID value; gather of 500 random feeds returns those cells [M] |
| P4b | `setup_hzcd` on a MiniSplit of 2,000 real feeds, sample keep file, 10 directories | log clause present; view == keep ? stored : fp16(transform(file)); popularity columns bit-identical to the 6b view of the base; `pre2_fe_s` == no-ID value on nc cells, stored bits elsewhere; every other key / column == plain gather; popularity-light list swaps `pre2_fe_z`, `pre2_fe_s` (+ `xp_imp_c50`); partial head refused; `cvnd_pre2` refused [M] |
| P4c | real Ranker (`w5_e10_kd05w192_s951@77500`), 600 feeds, 30 AdamW steps through `HzTrainView(cd_arms6c)` for `feats_n`, `feats_nP`, `feats_nF` | finite losses, weights move. Before any step, mean \|d within-feed z of the logit\| against the 6b hz2 view: 0.012 on all cells, 0.022 on nc cells; the hz2 view against stored inputs: 0.160 [M] |

P4 as asked ("a tiny cooldown with --hzcd through the harness, flag off must not change") is met only in part. `nn_x/w12/stale/fulldata/cpu_run.py` executes `train.py` with the whole train split in memory; I count about 45 GB resident and more than 60 GB at the peak of `attach_xt` [I], above the 40 GB I was given, so I did not run it. Flag-off identity rests on three facts: the import line is the only AST difference of `train.py` (35 top-level statements, 1 differs) [M]; `cd_arms6c` at import reads two environment variables and defines names, like 6b plus two constants [M]; without `--hzcd` no function of the module is called [D, `train.py` l.457-459, 491, 605-609].

## 4. Gate (label-free, all 246,745,836 train rows)

"pool" = trPool private-like feeds of days >= Sep 13 (the population of `hz2_review/pre2_tell.py`). A = all view-cold rows with a score, B = made cold and ID-carrying, C = made cold without ID, D = as-built cold. SMD = (mean on the train set - mean on the target's cold rows) / SD over all target rows.

```
TARGETS (as built) [M]: P(pre2 ID contribution != 0 | it_hist_all == 0): val 0 of 527423, public 0 of 379250, private 0 of 919809 cold rows; P(new-head ID | cold) 0.0000 / 0.0000 / 0.0000; P(seq_cv_pop > 0 | cold) 0.0000 / 0.0000 / 0.0000; pre2_fe_s cold mean (all-row SD): val 5.6331 (0.7515), public 5.6543 (0.7333), private 5.6802 (0.7353)

G1 / G3 (fe head). cold = effective view counter == 0. before -> after = P(fe-head ID contribution != 0 | cold), count after in (). rows chg = share of rows whose pre2_fe_s / pre2_fe_z bits change; feeds = share of feeds with a changed row.
dir        days  cold share G1 before       G1 after (count)  chg fe_s  chg fe_z   feeds
feats_n    all       0.4079    0.2357     0.000000 (     0)    0.2078    0.7198  0.7185
feats_n    >=13      0.4248    0.3020     0.000000 (     0)    0.2583    0.9986  0.9872
feats_n    pool      0.4572    0.2228     0.000000 (     0)    0.2207    1.0000  1.0000
feats_Ln   all       0.4232    0.2452     0.000000 (     0)    0.2231    0.7200  0.7202
feats_Ln   >=13      0.4605    0.3172     0.000000 (     0)    0.2941    0.9991  0.9912
feats_Ln   pool      0.4972    0.2414     0.000000 (     0)    0.2607    1.0000  1.0000
feats_hn   all       0.2276    0.1104     0.000000 (     0)    0.0686    0.6604  0.6067
feats_hn   >=13      0.2529    0.1348     0.000000 (     0)    0.0864    0.9031  0.8189
feats_hn   pool      0.3077    0.0875     0.000000 (     0)    0.0711    0.8937  0.8780
feats_nP   all       0.4079    0.2357     0.000129 ( 13001)    0.0962    0.7137  0.6823
feats_nP   >=13      0.4248    0.3020     0.000080 (  3388)    0.1283    0.9898  0.9359
feats_nP   pool      0.4572    0.2228     0.000165 (   478)    0.1019    0.9990  0.9977
feats_LnP  all       0.4232    0.2452     0.000129 ( 13502)    0.1038    0.7149  0.6874
feats_LnP  >=13      0.4605    0.3172     0.000084 (  3840)    0.1461    0.9926  0.9481
feats_LnP  pool      0.4972    0.2414     0.000177 (   559)    0.1200    0.9995  0.9989
feats_hnP  all       0.2276    0.1104     0.000064 (  3607)    0.0251    0.5430  0.4520
feats_hnP  >=13      0.2529    0.1348     0.000037 (   934)    0.0341    0.7460  0.6172
feats_hnP  pool      0.3077    0.0875     0.000060 (   118)    0.0269    0.7463  0.7192

G2 (fe head): standardised mean difference of pre2_fe_s vs PRIVATE cold rows (public in []), before -> after. A = all view-cold rows with a score, B = made cold AND ID-carrying, C = made cold, no ID, D = as-built cold (never touched).
dir        days  B                                  A                                  C                                  D               
feats_n    all   +0.641 -> +0.366 [+0.678 -> +0.403] +0.433 -> +0.334 [+0.470 -> +0.370] +0.411 -> +0.370 [+0.447 -> +0.407] +0.265 [+0.301]
feats_n    >=13  +0.585 -> +0.351 [+0.622 -> +0.387] +0.389 -> +0.311 [+0.425 -> +0.347] +0.376 -> +0.352 [+0.412 -> +0.389] +0.248 [+0.284]
feats_n    pool  +0.207 -> -0.003 [+0.243 -> +0.032] +0.029 -> -0.023 [+0.065 -> +0.012] +0.011 -> -0.011 [+0.046 -> +0.025] -0.038 [-0.002]
feats_Ln   all   +0.633 -> +0.369 [+0.670 -> +0.406] +0.438 -> +0.339 [+0.474 -> +0.376] +0.416 -> +0.377 [+0.453 -> +0.414] +0.265 [+0.301]
feats_Ln   >=13  +0.578 -> +0.357 [+0.615 -> +0.394] +0.398 -> +0.321 [+0.435 -> +0.357] +0.390 -> +0.368 [+0.427 -> +0.405] +0.248 [+0.284]
feats_Ln   pool  +0.203 -> +0.010 [+0.239 -> +0.045] +0.043 -> -0.010 [+0.079 -> +0.026] +0.042 -> +0.021 [+0.077 -> +0.057] -0.038 [-0.002]
feats_hn   all   +0.571 -> +0.367 [+0.608 -> +0.404] +0.322 -> +0.282 [+0.358 -> +0.318] +0.312 -> +0.271 [+0.348 -> +0.307] +0.265 [+0.301]
feats_hn   >=13  +0.528 -> +0.358 [+0.565 -> +0.394] +0.298 -> +0.270 [+0.334 -> +0.306] +0.309 -> +0.284 [+0.345 -> +0.320] +0.248 [+0.284]
feats_hn   pool  +0.117 -> -0.043 [+0.153 -> -0.007] -0.035 -> -0.052 [+0.000 -> -0.017] -0.115 -> -0.138 [-0.079 -> -0.103] -0.038 [-0.002]
feats_nP   all   +0.641 -> +0.400 [+0.678 -> +0.437] +0.433 -> +0.359 [+0.470 -> +0.396] +0.411 -> +0.411 [+0.447 -> +0.447] +0.265 [+0.301]
feats_nP   >=13  +0.585 -> +0.367 [+0.622 -> +0.404] +0.389 -> +0.323 [+0.425 -> +0.359] +0.376 -> +0.376 [+0.412 -> +0.412] +0.248 [+0.284]
feats_nP   pool  +0.207 -> +0.013 [+0.243 -> +0.048] +0.029 -> -0.014 [+0.065 -> +0.022] +0.011 -> +0.011 [+0.046 -> +0.046] -0.038 [-0.002]
feats_LnP  all   +0.633 -> +0.402 [+0.670 -> +0.438] +0.438 -> +0.364 [+0.474 -> +0.400] +0.416 -> +0.416 [+0.453 -> +0.453] +0.265 [+0.301]
feats_LnP  >=13  +0.578 -> +0.373 [+0.615 -> +0.410] +0.398 -> +0.333 [+0.435 -> +0.370] +0.390 -> +0.390 [+0.427 -> +0.427] +0.248 [+0.284]
feats_LnP  pool  +0.203 -> +0.025 [+0.239 -> +0.060] +0.043 -> -0.000 [+0.079 -> +0.035] +0.042 -> +0.042 [+0.077 -> +0.077] -0.038 [-0.002]
feats_hnP  all   +0.571 -> +0.400 [+0.608 -> +0.437] +0.322 -> +0.297 [+0.358 -> +0.333] +0.312 -> +0.312 [+0.348 -> +0.348] +0.265 [+0.301]
feats_hnP  >=13  +0.528 -> +0.374 [+0.565 -> +0.410] +0.298 -> +0.277 [+0.334 -> +0.314] +0.309 -> +0.309 [+0.345 -> +0.345] +0.248 [+0.284]
feats_hnP  pool  +0.117 -> -0.027 [+0.153 -> +0.008] -0.035 -> -0.048 [+0.000 -> -0.013] -0.115 -> -0.115 [-0.079 -> -0.079] -0.038 [-0.002]

G2 for pre2_fe_z (clipped +-5) vs PRIVATE cold rows, before -> after:
feats_n    all   B +0.184 -> -0.134 | A -0.038 -> -0.117 | C -0.102 -> -0.094 | D -0.177 -> -0.126
feats_n    >=13  B +0.165 -> -0.122 | A -0.054 -> -0.119 | C -0.108 -> -0.090 | D -0.181 -> -0.139
feats_n    pool  B +0.160 -> -0.082 | A -0.039 -> -0.077 | C -0.086 -> -0.083 | D -0.101 -> -0.073
feats_Ln   all   B +0.178 -> -0.124 | A -0.032 -> -0.107 | C -0.089 -> -0.079 | D -0.177 -> -0.123
feats_Ln   >=13  B +0.158 -> -0.108 | A -0.042 -> -0.103 | C -0.084 -> -0.062 | D -0.181 -> -0.135
feats_Ln   pool  B +0.149 -> -0.069 | A -0.028 -> -0.064 | C -0.055 -> -0.048 | D -0.101 -> -0.070
feats_hn   all   B +0.089 -> -0.183 | A -0.149 -> -0.191 | C -0.220 -> -0.258 | D -0.177 -> -0.164
feats_hn   >=13  B +0.075 -> -0.163 | A -0.150 -> -0.179 | C -0.198 -> -0.217 | D -0.181 -> -0.171
feats_hn   pool  B +0.065 -> -0.134 | A -0.101 -> -0.117 | C -0.202 -> -0.223 | D -0.101 -> -0.095
feats_nP   all   B +0.184 -> -0.103 | A -0.038 -> -0.098 | C -0.102 -> -0.056 | D -0.177 -> -0.140
feats_nP   >=13  B +0.165 -> -0.106 | A -0.054 -> -0.109 | C -0.108 -> -0.066 | D -0.181 -> -0.146
feats_nP   pool  B +0.160 -> -0.066 | A -0.039 -> -0.071 | C -0.086 -> -0.060 | D -0.101 -> -0.079
feats_LnP  all   B +0.178 -> -0.095 | A -0.032 -> -0.089 | C -0.089 -> -0.043 | D -0.177 -> -0.138
feats_LnP  >=13  B +0.158 -> -0.092 | A -0.042 -> -0.094 | C -0.084 -> -0.040 | D -0.181 -> -0.143
feats_LnP  pool  B +0.149 -> -0.055 | A -0.028 -> -0.057 | C -0.055 -> -0.027 | D -0.101 -> -0.076
feats_hnP  all   B +0.089 -> -0.143 | A -0.149 -> -0.175 | C -0.220 -> -0.208 | D -0.177 -> -0.169
feats_hnP  >=13  B +0.075 -> -0.143 | A -0.150 -> -0.172 | C -0.198 -> -0.186 | D -0.181 -> -0.173
feats_hnP  pool  B +0.065 -> -0.115 | A -0.101 -> -0.112 | C -0.202 -> -0.195 | D -0.101 -> -0.097

G4 (not repaired by the as-asked / P files): P(NEW-head ID contribution | cold), P(seq_cv_pop > 0 | cold), P(xp_imp_c50 > 0 | cold); targets 0 / 0 / 0.18-0.31
feats_n    all   new-head ID 0.2357 | seq_cv_pop > 0 0.0000 | xp_imp_c50 > 0 0.1179 | hmax32 ID-dependent 0.7505 (targets 0.98-0.99)
feats_n    >=13  new-head ID 0.3020 | seq_cv_pop > 0 0.0000 | xp_imp_c50 > 0 0.1467 | hmax32 ID-dependent 0.9851 (targets 0.98-0.99)
feats_n    pool  new-head ID 0.2228 | seq_cv_pop > 0 0.0000 | xp_imp_c50 > 0 0.1643 | hmax32 ID-dependent 0.9699 (targets 0.98-0.99)
feats_Ln   all   new-head ID 0.2452 | seq_cv_pop > 0 0.0000 | xp_imp_c50 > 0 0.1148 | hmax32 ID-dependent 0.7592 (targets 0.98-0.99)
feats_Ln   >=13  new-head ID 0.3172 | seq_cv_pop > 0 0.0000 | xp_imp_c50 > 0 0.1378 | hmax32 ID-dependent 0.9856 (targets 0.98-0.99)
feats_Ln   pool  new-head ID 0.2414 | seq_cv_pop > 0 0.0000 | xp_imp_c50 > 0 0.1570 | hmax32 ID-dependent 0.9709 (targets 0.98-0.99)
feats_hn   all   new-head ID 0.1104 | seq_cv_pop > 0 0.0971 | xp_imp_c50 > 0 0.3993 | hmax32 ID-dependent 0.7405 (targets 0.98-0.99)
feats_hn   >=13  new-head ID 0.1348 | seq_cv_pop > 0 0.0967 | xp_imp_c50 > 0 0.4435 | hmax32 ID-dependent 0.9823 (targets 0.98-0.99)
feats_hn   pool  new-head ID 0.0875 | seq_cv_pop > 0 0.0821 | xp_imp_c50 > 0 0.4579 | hmax32 ID-dependent 0.9663 (targets 0.98-0.99)

F family, NEW head: G1n = P(new-head ID contribution | cold) before -> after (count); rows chg; SMD of pre2_new_s vs private [public] before -> after for B and A
feats_nF   all   G1n 0.2357 -> 0.000000 (18) | chg new_s 0.0962 new_z 0.7137 | B +0.582 -> +0.377 [+0.622 -> +0.416] | A +0.403 -> +0.340 [+0.442 -> +0.379] | C +0.388 | D +0.253
feats_nF   >=13  G1n 0.3020 -> 0.000000 (7) | chg new_s 0.1283 new_z 0.9898 | B +0.533 -> +0.345 [+0.573 -> +0.384] | A +0.361 -> +0.304 [+0.400 -> +0.343] | C +0.352 | D +0.235
feats_nF   pool  G1n 0.2228 -> 0.000000 (0) | chg new_s 0.1019 new_z 0.9990 | B +0.150 -> -0.017 [+0.189 -> +0.021] | A +0.005 -> -0.032 [+0.043 -> +0.006] | C -0.011 | D -0.049
feats_LnF  all   G1n 0.2452 -> 0.000000 (20) | chg new_s 0.1038 new_z 0.7149 | B +0.577 -> +0.379 [+0.617 -> +0.419] | A +0.408 -> +0.345 [+0.447 -> +0.384] | C +0.394 | D +0.253
feats_LnF  >=13  G1n 0.3172 -> 0.000000 (9) | chg new_s 0.1461 new_z 0.9926 | B +0.530 -> +0.353 [+0.570 -> +0.392] | A +0.372 -> +0.315 [+0.411 -> +0.354] | C +0.369 | D +0.235
feats_LnF  pool  G1n 0.2414 -> 0.000000 (0) | chg new_s 0.1200 new_z 0.9995 | B +0.152 -> -0.003 [+0.190 -> +0.035] | A +0.019 -> -0.018 [+0.057 -> +0.020] | C +0.021 | D -0.049
feats_hnF  all   G1n 0.1104 -> 0.000000 (5) | chg new_s 0.0251 new_z 0.5430 | B +0.515 -> +0.371 [+0.555 -> +0.410] | A +0.298 -> +0.277 [+0.338 -> +0.316] | C +0.282 | D +0.253
feats_hnF  >=13  G1n 0.1348 -> 0.000000 (1) | chg new_s 0.0341 new_z 0.7460 | B +0.479 -> +0.347 [+0.518 -> +0.386] | A +0.277 -> +0.260 [+0.316 -> +0.298] | C +0.281 | D +0.235
feats_hnF  pool  G1n 0.0875 -> 0.000000 (0) | chg new_s 0.0269 new_z 0.7463 | B +0.065 -> -0.058 [+0.103 -> -0.020] | A -0.052 -> -0.062 [-0.014 -> -0.025] | C -0.138 | D -0.049

View c co-visitation (gate_g4_extra.json): P(x > 0 | view cold) / P(x > 0 | made cold): seq_cv_invd20 0.0097 / 0.0227; seq_cv_pop 0.0971 / 0.2269; seq_cv_norm 0.0097 / 0.0227; xp_imp_c50 0.3993 / 0.4949; it_hist_et1_luo 0.0000 / 0.0000; it_hist_et4_luo 0.0000 / 0.0000 | targets: seq_cv_* 0.0000 exactly; xp_imp_c50 val 0.305 public 0.218 private 0.185
```

Verdicts against the rules of `DECLARATION.txt`:

- **G1 PASS** for a, b, c: 0 view-cold rows keep an fe-head ID contribution; val / public / private measure 0 as built, so the premise holds.
- **G2(i)**: pool, B rows, `pre2_fe_s` against private cold rows: +0.207 -> -0.003 (a), +0.203 -> +0.010 (b), +0.117 -> -0.043 (c): PASS. All train feeds: +0.641 -> +0.366 (a), +0.633 -> +0.369 (b), +0.571 -> +0.367 (c): FAIL by the letter (the rule asked for less than half). Cause [M]: D rows, which no view touches, read +0.265 on all feeds and -0.038 on the pool; the all-feeds population (mostly production-ordered feeds) is not exchangeable with test feeds. After the change B sits at the level of the other cold rows of its own population (C +0.370, A +0.334). The declaration listed "G2(i) FAIL" as a reason not to train; I report that it fired on this reading and that I regard the criterion, not the view, as wrong.
- **G2(ii)**: all feeds PASS everywhere; pool PASS for a, b; pool FAIL for c (-0.035 -> -0.052 against private, +0.000 -> -0.017 against public), a move of 0.017 SD away from the target.
- **G3**: `pre2_fe_s` changes on 20.8% / 22.3% / 6.9% of all train rows (25.8% / 29.4% / 8.6% on days >= Sep 13); `pre2_fe_z` on 72.0% / 72.0% / 66.0% (99.9% / 99.9% / 90.3%); feeds with a changed row 71.9% / 72.0% / 60.7% (98.7% / 99.1% / 81.9%). Feeds of Sep 6-8: 0 rows change.
- **G4**: section 5.

## 5. Findings beyond the task

### 5.1 The new encoder head carries the same tell (not repaired as asked; repaired in F)

- P(raw new-head score != raw new-head no-ID score | view cold) = 0.236 / 0.245 / 0.110 (a / b / c, all days), 0.302 / 0.317 / 0.135 on days >= Sep 13; 0.0000 on cold val / public / private rows [M, `gate_full.json`]. In every train window the set of rows with a new-head ID contribution equals the set with `pre2_id_contrib` != 0 (0 rows in the symmetric difference) [M, `build_trainF.json`].
- On val / public / private the two sets are identical as well: 628,350 / 373,332 / 810,938 rows, 0 rows with one head and not the other; cold rows have neither [M, `head_mismatch.json`].
- The witness `both_pre2both` replaces the fe head only (its docstring says so [D]). It feeds 43,419 val rows (3.1% of val rows, 6.4% of its cold rows) with "fe head replaced, new head still with ID" [M]. `both_pre2full` replaces the new head on exactly those 43,419 rows.
- Consequence for training: the as-asked view turns "cold counters + both heads warm" into "cold counters + fe head cold-like + new head warm" on 23.6% / 24.5% / 11.0% of view-cold rows. That pattern is on no test row. An as-asked model is trained on the witness's own artefact, an F model on the test pattern.
- F files: built on the same nc rows exactly as `build_pre2full` (the new key's own affine map applied to the raw no-ID score). After: 18 / 20 / 5 rows where the no-ID value coincides bitwise with the stored one; `pre2_new_s` on B rows, pool: +0.150 -> -0.017 (a), +0.152 -> -0.003 (b), +0.065 -> -0.058 (c) [M, `gate_new.json`].
- `pre2_hmax32` is not touched: it differs from its no-ID twin on 98-99% of cold val / test rows and on 97-99% of view-cold train rows of days >= Sep 9 alike (history-item IDs). A candidate-ID part cannot be separated with the files at hand.

### 5.2 View c: cold counter with warm co-visitation (not repaired as asked; repaired in `feats_hnFS`)

- On val / public / private, `it_hist_all` = 0 implies `seq_cv_invd20` = `seq_cv_pop` = `seq_cv_norm` = 0 on all 527,423 / 379,250 / 919,809 cold rows [M, `gate_g4_extra.json`]. Views a and b keep this. View c swaps only the five `it_hist` columns: P(`seq_cv_pop` > 0 | view cold) = 0.097 (0.227 on the rows it makes cold; 5,453,915 rows), `seq_cv_invd20` and `seq_cv_norm` 0.0097 (545,443 rows). Arm hz was trained with this.
- `feats_hnFS`: the three files = where(view counter == 0, 0, stored); 0 rows with a positive view counter change; 0 view-cold rows stay positive [M, `build_seq0.json`]. No module change is needed (`cd_arms6b` already swaps these names when the file exists).
- `xp_imp_c50` > 0 on 40% of view-cold rows under view c against 18-31% on the targets: a level difference, left alone.

### 5.3 Train calibration of the no-ID score (as asked: inherited; P: removed)

- `pre2_fe_noid` - `pre2_fe` on rows with `pre2_id_contrib` = 0: -0.257 (SD 0.015) in window 1, -0.184 in window 2, -0.008 in window 3; val +0.0031, test 0 exactly [M, `gate_offset.json`]. Cause [D, `pretrain/assemble_train.py`]: the two keys are moment-matched separately to the full-corpus twin (window 1: v3 fe 1.1275 x + 1.801, fe_noid 1.133 x + 1.501). The review's "within 0.028" is the maximum of the last window.
- Effect: under "as asked" a made-cold row WITHOUT ID also moves, by -0.043 / -0.031 / -0.001 on the `pre2_fe_s` scale. Visible in the gate as C: +0.376 -> +0.352 (a, days >= Sep 13).
- P family: per config, the fe key's own map applied to the raw no-ID score; made-cold rows without ID keep their bits. On val it differs from the witness by at most 0.009 / 0.020 / 0.0015 (not witness-exact, as declared). Level against private cold rows, pool: A -0.023 (as asked) against -0.014 (P) for a; -0.010 against -0.000 for b; -0.052 against -0.048 for c. By my rule of ADDENDUM 1 the as-asked family stays the primary one; P is a sensitivity variant.
- P leaves 13,001 / 13,502 / 3,607 view-cold rows with a residual above 1e-3 against `pre2_fe` - `pre2_id_contrib` (0.013% of cold rows). 8,791 / 9,082 / 2,107 of them are rows where the assembler's guard "raw score > -2" differs between the score and its no-ID twin [M]; the remainder was not examined.

### 5.4 Unchanged by design

Warm rows of views a / b are 1-2 d stale instead of 7-10 d and 44.5% of warm values are non-integer [D, review MAJOR 1]; the Sep 6-8 keep hybrid fails the hz2 gate [D, review MEDIUM 3]. Nothing in this arm touches either.

## 6. Reasons not to train, per view

- (a) `feats_n`: none from the proofs. G2(i) fired on the all-feeds reading (section 4). Half of the encoder tell stays (5.1).
- (b) `feats_Ln`: no base arm exists (no hz2L checkpoint in `nn_x/w5/ckpt` at 07:43Z [M]); the base view fails the hz2 study's gate by G1 [D]. An hz2Ln arm read against hz2 mixes "lag-first" with the encoder change. Train it only next to an hz2L base arm.
- (c) `feats_hn`: none from the proofs; keeps the tells of 5.1 and 5.2 and fails G2(ii) on the pool by 0.017 SD.
- F and `feats_hnFS`: no reason from proofs or gate. They need the `both_pre2full` scenario for a like-for-like stale-val read.
- P: not recommended for GPU (5.3).

## 7. Limits and what I did not check

- `train.py` + `cd_arms6c` was not executed end to end (section 3). I started no GPU process; `stale_pred_full.py` and the scoring part of `hold_infer_n.py` were not run by me. The lead started `stale_pred_full.py` at 07:41Z (pid 178736, not mine, not touched); the first line of its log lists `pre2_new_s`, `pre2_new_z` among the overridden inputs and gives "base check vs stored: corr 1.0000000 max|d| 0.00e+00" [D, `preds_stale_full/stale_pred.log`; I read that part of the line only].
- No labelled number: the size of any effect on NDCG is unknown. P4c is a smoke of the code path on CPU fp32 with a plain AdamW step, not the trainer's optimiser or loss mix.
- The alignment constants of the train windows are re-derived from the 60,000-feed samples exactly as `assemble_train.py` does; P0 proves the stored columns come back bitwise, which covers the fe, fe_noid and new keys.
- G2 uses means and SDs only; I did not compare distributions beyond the first two moments, nor by listing age.
- Which no-ID calibration is closer to a test row (5.3) is not decided by the evidence here.
- `hz2n_newly_cold.npy` uses the raw view file; a model whose xt list has no `it_hist` column (popularity-light) still gets the encoder swap on those rows, by design.
- Memory: peak resident set of `gate.py` 25.1 GB and of `p4a_full.py` 17.9 GB [M, `/usr/bin/time -v`, `logs/gate_memtest.time`, `logs/p4a_memtest.time`]; `build_enc.py train` about 20 GB [I, from array counts, not measured: re-running it would rewrite delivered files].

## 8. Pre-registration (written 2026-11-15T07:37:51Z, before any hz2n run; the lead's lane arms were added at 07:41Z)

```
PRE-REGISTRATION of the labelled read of arm hz2n. Written 2026-11-15T07:37:51Z by the builder, BEFORE any hz2n run exists (no GPU job of this arm has been started;
the builder has opened no val / test / holdout label). These are EXPECTATIONS [I], not results. Same read as hz / hz2: 16-pool arm minus 16-pool of the
comparison arm at the last step (@last), paired per feed over the 14,423 val feeds; stale-val = nn_x/w5/stale_pred.py (scenario both_pre2both).

Basis of the expectation: (1) the encoder swap is a small change in output space next to the popularity view itself: with the re-cool init checkpoint, before any
step, mean |d within-feed z of the logit| = 0.011 over all cells and 0.022 on newly cold cells for "feats_n minus the 6b view of hz2/feats", against 0.16 for
"hz2 view minus stored inputs" [M, logs/test_cd_arms6c_p4c.log]; (2) removing the item-ID part of the encoder score at INFERENCE costs 0.00000-0.00005 on as-built
models [D, hz2_review.md MAJOR 2]; (3) the tell is label-relevant in training (within-feed AUC 0.528 on hz2-cold rows [D, same]); (4) paired SE of a 16-pool
difference between two hz2-type arms is about 0.00047 [D, E_kd_predeclared.txt 07:15Z, row hz2 - hz].

 arm (HZCD_DIR)                      | comparison | val: expected [80% interval]      | stale-val (both_pre2both): expected [80% interval] | in-period holdout (reported only)
 hz2n  as asked (feats_n)            | hz2        |  0.0000 [-0.0006, +0.0006]        | +0.0002 [-0.0005, +0.0009]                         | -0.0002 [-0.0008, +0.0004]
 hz2n  as asked (feats_n)            | ref        | +0.0003 [-0.0006, +0.0012]        | +0.0016 [+0.0006, +0.0027]                         | -0.0030 [-0.0040, -0.0020]
 hz2nF both heads (feats_nF)         | hz2        |  0.0000 [-0.0006, +0.0006]        | +0.0001 [-0.0006, +0.0008]                         | -0.0003 [-0.0010, +0.0004]
 hzn   as asked (feats_hn)           | hz         |  0.0000 [-0.0005, +0.0005]        | +0.0001 [-0.0005, +0.0007]                         | -0.0001 [-0.0005, +0.0003]
 hznFS (feats_hnFS)                  | hz         |  0.0000 [-0.0005, +0.0005]        | +0.0002 [-0.0005, +0.0009]                         | -0.0002 [-0.0007, +0.0003]
 hz2Ln as asked (feats_Ln)           | hz2        |  0.0000 [-0.0007, +0.0007]        | +0.0002 [-0.0006, +0.0010]                         | -0.0006 [-0.0015, +0.0003]

Plain statement: I expect NO arm of this family to separate from its base arm (hz2 or hz) by more than 1.5 SE on val or on stale-val. The most likely outcome is
a null against the base arm with the base arm's advantage over ref kept. A result outside an interval above is information about my model of the effect, not a
success criterion; the lead's RULE hz governs the arm.
Asymmetry to keep in mind when reading stale-val (not a prediction): the witness scenario both_pre2both replaces the fe head only. The "as asked" arms are trained on
exactly that pattern (fe head without ID, new head with ID on newly cold rows); the F arms are trained on the test pattern (no ID in either head). So both_pre2both
is biased toward "as asked" and against F by construction; the like-for-like witness for F is both_pre2full (stale_pred_full.py in this directory, GPU, not run by me).
My expectation for F minus "as asked" (same base view): both_pre2both stale-val -0.0001 [-0.0007, +0.0005]; both_pre2full stale-val +0.0002 [-0.0004, +0.0008].
```

## 9. Commands

```
cd <repo>/nn_x/w12/stale/hz2n
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
PY=<repo>/.venv/bin/python
$PY build_enc.py val            # 4 s    P2            -> p2_val_parity.json
$PY build_enc.py train          # 80 s   P0, P3, files -> feats_{n,Ln,hn}/train/pre2_hz_fe*.npy, build_train.json
$PY gate.py offset              # 10 s   offset of the no-ID score -> gate_offset.json
$PY build_enc.py val P ; $PY build_enc.py train P     # P family -> feats_{n,Ln,hn}P, build_trainP.json
$PY build_new.py val ; $PY build_new.py train         # F family: P2n, P0n, P3n -> feats_{n,Ln,hn}F, build_trainF.json
$PY build_seq0.py               # 8 s    feats_hnFS
$PY make_6c.py                  # cd_arms6c.py + diff
$PY test_cd_arms6c.py p1 p4b p4c   # 2 min -> logs/test_cd_arms6c_{p1,p4b,p4c}.log
$PY p4a_full.py ; $PY p4a_full.py feats_hnFS          # 7 + 1 min -> p4a_full*.json, logs/p4a_full*.log
$PY gate.py full ; $PY gate_new.py ; $PY gate_table.py   # 4.5 + 1 min -> gate_full.json, gate_new.json, gate_table.txt
$PY hold_infer_n.py selftest    # 25 s
$PY make_ready.py ; $PY make_report.py
# moving the has: flag to aux/, the base-view symlinks and gate_g4_extra.json / head_mismatch.json were made by inline commands recorded in this session's log; the symlink
# rule is: every file of the base train directory for a / b, the five it_hist_hz_* files for c.
```

## 10. Files

All under `nn_x/w12/stale/hz2n/`: `READY.txt`, `DECLARATION.txt`, `PREREG_hz2n_arm.txt`, `PROGRESS.md`, `MD5SUMS.txt`, `cd_arms6c.py` (md5 `dbe5a1387569ecbcb832de6e4cc97297`), `make_6c.py`, `cd_arms6c_vs_cd_arms6b.diff`, `train_py_import_6c.txt`, `enc_common.py`, `build_enc.py`, `build_new.py`, `build_seq0.py`, `gate.py`, `gate_new.py`, `gate_table.py`, `gate_table.txt`, `test_cd_arms6c.py`, `p4a_full.py`, `stale_pred_full.py`, `hold_infer_n.py`, `make_ready.py`, `make_report.py`, the json results (`build_train*.json`, `p2*_val_parity*.json`, `gate_*.json`, `p4a_full*.json`, `build_seq0.json`, `head_mismatch.json`), `logs/`, `cache/` (test fixtures), and the ten view directories. Report: `reports/STALE_1115/hz2n.md`.

## 11. Self-report on the rules

- Scope: I built more than the three views (families P and F, `feats_hnFS`, two reader wrappers). Each was declared with a time stamp before its first number; the as-asked files were not altered by them.
- Declaration gaps: the P family's "G1 after" reading (a 1e-3 tolerance against `pre2_fe` - `pre2_id_contrib`) was fixed in `gate.py` before the P files were gated but is not in `DECLARATION.txt`. The G2(i) population was ambiguous in my text ("all feeds" in the header, "the review's population" in the clause); I report both readings.
- Labels: no val / test / holdout label file was opened. `MiniSplit` loads `cand_target` of the sampled TRAIN feeds (holdout feeds excluded from every sample); those train labels were used only as the loss target of the 30-step smoke P4c. No choice depends on them.
- Writes: only inside `nn_x/w12/stale/hz2n/` and this report. I read `nn_x/w5/lane.sh`, `train.py.6c` and the lead's rule file at 07:43Z without touching them.
- Processes: CPU only; at most two heavy processes at a time, 12 threads each; all started by me have ended. I signalled no process. Two single foreground `sleep` calls (1 s and 20 s) were used to let a background job start; no chains.
- Not met: the CPU run of `train.py` (memory limit), see section 3.
