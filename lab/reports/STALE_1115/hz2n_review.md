# hz2n — adversarial review

Reviewer session 2026-11-15T07:48:56Z – 08:16Z. CPU only. Directory `nn_x/w12/stale/hz2n_review/` (DECLARATION.txt, PROGRESS.md, scripts, JSON outputs).
Marks: [M] measured by me with my own code, [D] read in a project file, [I] inferred. Builder's files were read; none of its functions were used for a check
(one exception, declared: `enc_common.mix_rows` was run next to mine on val, to compare the two).

## 0. Verdict

**CONFIRMED: safe to train, safe to use in a final member.** The build does exactly what it states, nothing leaks, and the gate numbers hold.

| item | result |
|---|---|
| 1 identity 6c vs 6b | PASS: text, AST, behaviour bit-identical without the new files [M] |
| 2 witness parity val | PASS: 0 of 1,389,971 rows differ bitwise (fe, fe_s); z max\|d\| 4.6e-6 (float64 vs float32) [M] |
| 2 witness rule on train | PASS on ALL rows, not a sample: 246,745,836 rows row-wise relations, 177,782,868 window rows from raw scores [M] |
| 3 leak audit | CLEAN: no path from a row's label, a later same-user snapshot or a val / test / holdout label to any new value |
| 4 gate recount | the builder's G1 / G2 / G3 "before" and "after" numbers all reproduce [M] |
| 5 runs | 4 GPU runs `w5_kcd_hznFS_s951-954` (started 07:52:24Z) print the setup line I obtained independently on CPU, to the digit [M] |

What the verdict does NOT say: the family is not "every tell removed". One same-family combination that is impossible at test remains in the views
(section 5, MEDIUM, not introduced by hz2n, not seen by the builder's gate), and the stale-val witness has the same hole.

## 1. Identity (item 1)

- `r1_ast.py` [M]: md5 `nn_x/w5/cd_arms6c.py` == builder's copy (dbe5a138…). Changed lines lie only in: module docstring, two new constants `ENC_TO_HZ` / `ENC_GROUPS`
  (+ 4 comment lines), `luo_cols`, `setup_hzcd`. Top-level statement order identical; every other def (`hz_path`, `xt_transform`, `load_hz`, `apply_keep`, `HzTrainView`,
  `hz_stats`, `save_holdout_hz`) is `ast.dump`-identical.
- Changed statements, each explained:
  - `luo_cols` l.85-86: list of encoder columns whose file exists in `<HZCD_DIR>/train/`; empty when no `pre2_hz_*` file exists.
  - `luo_cols` l.87-91: two asserts inside `if enc:` (partial head refused; preset with `pre2_id_contrib` refused). Dead when `enc` is empty.
  - `luo_cols` l.92: `cols += enc` (no-op when empty). After deleting these three statements the function is AST-identical to 6b.
  - `setup_hzcd` l.192-194: builds the log clause. l.199: tag `6b` -> `6c` and `{enc_msg}` in the f-string. Nothing else differs (313 vs 316 AST nodes, all in that f-string).
- Behaviour without new files (`r1c_one.py`, own stand-in split, 2,476 own random feeds = 198,626 rows, all day slots incl. Sep 6-8) [M]:
  environments unset / hz2 feats + keep / hz2 feats_L + keep, xt lists cvnd_pre2s and a popularity-light list: `luo_cols`, `load_hz` tensor, `apply_keep`, 39 gathered batches
  (every key), `hz_stats`: sha256 equal in all six comparisons. The popularity-light list with HZCD_DIR unset raises the same assert in both modules.
- Transform parity (`r1d_transform.py`) [M]: `xt_transform(hz name)` == the `attach_xt` branch of the stored name on 4,000,000 real rows (1,000,000 NaN rows) for all 16 mapped names, 0 differing bits.
- Behaviour change to know (LOW): with a directory that holds `pre2_hz_*` files, a model WITHOUT any LUO counter input (popularity-light list) no longer hits the
  "no LUO popularity column" assert; it trains with only the encoder columns swapped (feats_hn: `[pre2_fe_z, pre2_fe_s]`). Under view c `xp_imp_c50` is not swapped:
  on rows made cold P(xp_imp_c50 > 0) = 0.64 vs 0.18 on private cold rows [M, n6_scan_hnFS]. Do not re-cool popularity-light students through a view-c directory without deciding this first.

## 2. Witness parity (item 2)

- Val (`r2a_val.py`, numpy, float64 per-feed moments) [M]: nc == `newly_cold.npy`; `pre2both_pre2_fe` / `_s`: 0 rows differ bitwise; `_z`: max|d| 4.6e-6, no degenerate feed.
  Power controls: z of the unmixed scores differs by up to 2.29, naive z of the config mean by 0.37. Builder's function on the same inputs: 0 / 0 / 0 rows differ vs the witness.
- Train, row-wise relations, ALL rows, views a / b / c (`r2b_full.py`) [M], violations in every line = 0:
  builder's `hz2n_newly_cold` == my nc; `pre2_hz_fe` == where(nc, noid, fe) bitwise; `pre2_hz_fe_s` bitwise stored on non-nc rows and bitwise float32((noid+20)/6) on nc rows;
  has file == ~isnan(new score) == ~isnan(stored score) (0 flags change); keep rows: 0 changed in any file, 0 of 46,918,534 keep rows carry a score;
  `pre2_hz_fe_z` bitwise stored in every feed without an nc row. F files: 0 rows changed outside nc, 0 keep rows changed. FS files: bitwise where(V == 0, 0, stored), 0 positive on cold rows.
- Train from RAW window scores, all 177,782,868 window rows (`r2c_raw.py`) [M]: my alignment reproduces stored `pre2_fe`, `pre2_fe_noid`, `pre2_new` bitwise (0 rows);
  builder's `pre2_hz_fe` and `pre2_hz_new` == my recomputation bitwise in all three views and windows; `pre2_hz_fe_z` max|d| 1.7e-5, `pre2_hz_new_z` 2.5e-5, 0 rows above 1e-4.
  Per-feed standardisation: feed means of the z file <= 1.7e-5, feed SD median 0.993 (config average), i.e. recomputed over all M rows of the mixed per-config score. Control: unmixed z differs by up to 6.96.
- Shares changed agree with READY.txt to the printed digit (fe_s 0.2078 / 0.2231 / 0.0686; fe_z 0.7198 / 0.7200 / 0.6604; feeds 0.7185 / 0.7202 / 0.6067).
- Deviation from my declaration: the declared 1,500-feed samples were replaced by all rows (stronger).

## 3. Leak audit (item 3) — CLEAN

Inputs of every new value: view counter V, stored counter S, keep mask, per-config window scores fe / fe_noid / new / new_noid, twelve calibration moments per window.
- V, S, keep: label-free by their builders [D `feat_new/it_hist_hz.py`, `hz2/views.py`]; same-user later snapshots are excluded for non-zero users (the LUO rule).
- `pre2_fe_noid`: `pretrain/train.py` infer() l.239-254, the same forward pass as `fe` with the candidate's ID slot zeroed (`item_rep(..., zeros)`) and without the ID bias; same query vector.
  One function for train windows, val and test. Window corpora hold, per user, the latest pre-window snapshot and the earliest in-window one [D `build_corpus_window.py`]: no later same-user snapshot.
- Alignment differs by split, and this matters (section 5.2): train = moment match per key to the full-corpus twin on a 60,000-feed sample; val = affine per key to the test-side twin; test = none.
  The twin saw later snapshots, but only two moments per key / config / window enter (no row-level information).
- Standardisation: within-feed z over the feed's own M rows; `(x + 20) / 6` is fixed. No statistic spans feeds with labels.
- No val / test / holdout label file is opened by the builder's scripts I read or by mine. Disclosure: the "pool" population (to recount the builder's pool columns) is the project's
  definition via `train_prandom.npz`, which is derived from train click positions; it defines a population only.

## 4. Gate recount (item 4), full arrays [M]

| view | days | cold share | G1 before -> after | B pool: SMD vs private cold | A pool |
|---|---|---|---|---|---|
| a feats_n | all / >=Sep13 / pool | 0.4079 / 0.4248 / 0.4572 | 0.2357 / 0.3020 / 0.2228 -> 0 rows | +0.207 -> -0.003 | +0.030 -> -0.023 |
| b feats_Ln | same | 0.4232 / 0.4605 / 0.4972 | 0.2452 / 0.3172 / 0.2414 -> 0 rows | +0.203 -> +0.010 | +0.043 -> -0.010 |
| c feats_hn | same | 0.2276 / 0.2529 / 0.3077 | 0.1104 / 0.1348 / 0.0875 -> 0 rows | +0.117 -> -0.043 | -0.035 -> -0.053 |

All feeds: B +0.641 -> +0.366 (a), +0.633 -> +0.369 (b), +0.570 -> +0.367 (c); untouched cold rows D +0.265 (all feeds), -0.038 (pool). Every number is within 0.005 of `gate_table.txt`.
Model-input form of G1 (pre2_fe_s differs from the no-ID value on an ID-carrying cold row): 0 rows in all views.
- I agree with the builder on G2(i): the "all feeds" FAIL is a population offset (D rows read +0.265 there), not a defect of the view.
  Offset-free form: B minus E (made cold with vs without ID) +0.196 -> +0.008 (a, pool), +0.231 -> -0.004 (a, all feeds).
- View c, level evidence is neutral: pre2_fe_s gap of B rows to private cold +0.117 -> -0.043 (closer), pre2_fe_z gap +0.065 -> -0.133 (further; private cold mean z -0.119, B -0.054 -> -0.252).
  The change puts B rows at the level of the view's other cold rows (D -0.213, E -0.341), not closer to private. The arm's value has to come from the witnesses.
- Builder findings 3 and 4 confirmed: P(new-head ID | view cold) == P(fe-head ID | view cold) in every window (same rows, xor count 0); after F 18 / 20 / 5 coincidental rows.
  View c: P(seq_cv_pop > 0 | view cold) 0.0971 (0.2269 on rows made cold), 0 after FS.
- Generic scan of all 34 model inputs through feats_nF and feats_hnFS (`n6_scan.py`): on view-cold rows no input has a positive-rate identity broken (target exactly 0 or 1).

## 5. Issues

### 5.1 MEDIUM — the ID threshold is 3, the view repairs counter == 0 only

P(pre2 ID contribution != 0 | it_hist_all in {1, 2}) is exactly 0 on val, public, private (0 of 244,563 private rows) and as-built train rows: the encoder's ID slot needs >= 3 occurrences.
Through the views it is not (`n8_lowcount.py`, `n8b_lowrows.py`, samples of 30,000 feeds) [M]:

| | P(ID \| counter in (0,1]) | (1,2] | L rows, share of rows (pool / >=Sep13) | L rows / B rows fixed by hz2n | ID-contribution-weighted |
|---|---|---|---|---|---|
| private / val / as built | 0 | 0 | 0 | | |
| view a (hz2, hz2n) | 0.73 / 0.80 | 0.28 / 0.35 | 4.1% / 4.7% | 0.41 / 0.36 | 0.47 / 0.46 |
| view c (hz, hzn*) | 0.14 / 0.21 | 0.10 / 0.17 | 1.5% / 2.3% | 0.57 / 0.69 | 0.96 / 1.13 |

L rows = view counter in (0, 3) with an ID-carrying encoder score. Their mean ID contribution is larger than on the repaired rows (+1.06 vs +0.63 on the fe scale in view c, pool).
For view c, the recipe X view, what remains is about as large as what hz2n removes. hz, hz2 and every hz2n directory carry it; the builder's G1 conditions on counter == 0 and cannot see it.
- Not declared before its first number (found by the scan; ADDENDUM R-1 in my DECLARATION.txt). By my declared index, days >= Sep 13, against view a's original defect:
  view a 0.17 by the letter and 0.43 offset-corrected (MEDIUM on both); view c 0.08 by the letter (LOW) and 0.19 offset-corrected (MEDIUM).
  Against the view's own repaired defect it is 0.46 (a) and 1.13 (c). I file it as MEDIUM.
- The witness has the same hole: under scenario 'both', 51,730 val rows (3.72%) have counter 1-2 with an ID-carrying score in both heads, more than the 43,419 rows (3.12%)
  that pre2both / pre2full repair. hz / hz2 were trained on such rows, the reference was not, so SV and SVF may flatter the arms on those rows. Size unknown: NOT PROVEN either way.
- Level evidence for repairing L rows is mixed: view a +0.185 -> about -0.04 SD vs private low-counter rows [I], view c +0.107 -> about -0.13 [I].
- This does not block training: every hz2n directory is closer to test than its base on every count I made.

### 5.2 LOW (MEDIUM on the offset-corrected reading) — rows made cold WITHOUT an ID are shifted too

Builder's finding 5, confirmed: train `pre2_fe_noid` sits below `pre2_fe` on rows without any ID by -0.2571 / -0.1841 / -0.0084 (windows Sep 9-11 / 12-15 / 17-20; val +0.003, test 0),
because the two keys have separate moment matches. The as-asked rule therefore moves E rows (made cold, never had an ID): 11.2% of all rows in view a, 4.3% in view c, by -0.058 / -0.042 / -0.002 target SD.
At test such rows have fe == noid exactly. Index vs the original defect: 0.04 by the letter of my rule, 0.10-0.12 offset-corrected (view a); against view c's own repaired defect 0.22-0.35 (0.52 in window Sep 9-11).
In F directories the new head does not have this shift (its own map is used), so the two heads are treated differently on E rows in windows 1-2.
The P family removes it for the fe head; a P-headed F / FS directory needs only symlinks. I did not verify the P files.

### 5.3 LOW — hmax32: right conclusion, wrong argument

`pre2_hmax32` is built with the candidate's ID-aware representation (`pretrain/train.py` l.256-263), so on B rows it carries the candidate ID while the counter says cold.
"P(hmax32 != hmax32_noid | cold) is alike on train and test" does not address this (the noid variant also strips history IDs). My estimate of the candidate-ID effect on B rows
(B minus E in hmax32 - hmax32_noid): -0.01 to -0.03 SD in every window and config (`n5_hmax.py`) [M]. Negligible; leaving it untouched is fine.

### 5.4 LOW — remaining items
- P4 of the builder (train.py end to end) is now covered: the four GPU runs print 12 swapped / 4 encoder / keep 0 and per-column "rows changed" 0.0022, 0.0221, 0.0022, 0.1669, 0.1606, 0.0827,
  0.0821, 0.0603, 0.6199, 0.5141, 0.0626, 0.0248, identical to my CPU run of the real `setup_hzcd` over all rows (`r5_setup_full.py`; 0 differing cells vs my independent expectation) and to READY.txt 5b.
  For hz2nF expect 12 / 4 / keep 46,918,534 and 0.1272, 0.4843, 0.1297, 0.6809, 0.6570, 0.3911, 0.3834, 0.3183, 0.7172, 0.7095, 0.1914, 0.0952 [M].
- train.py flag-off path: current md5 818f4537…, backup `train_pre_6c.py` e5391fb8…; only line 56 differs; AST identical after normalising the module name; the two imported names are used
  only at l.459 and l.606, both behind `if a.hzcd`. `nn_x/w12/kd/fd/train_full2.py` still imports 6b (correct for the X:hz full-data re-cools; it cannot read an hz2n directory's encoder files).
- View a is also off on the warm side: P(ID | view-warm) 0.86 vs 0.74 private and 0.745 as built (view c 0.78). Property of the base view, unchanged by hz2n.
- The hz2n directory has no LIVE marker; four runs now read symlinks into it. MD5 of the eleven files the two arms read equal MD5SUMS.txt at 08:12Z [M].

## 6. What I recommend

1. Let hznFS run and launch hz2nF as planned; read them with the rule of 07:42Z. Nothing found here argues against it.
2. Do not describe hznFS as free of train-only combinations. It is free of them on cold rows only.
3. Before deciding recipe X on an hz2n arm, one cheap robustness read of the witness: I built the val files for a scenario `both_pre2lt3` (no-ID score in both heads also where the 'both' counter is 1-2
   and the row carries an ID; 51,730 more rows) and a wrapper in the pattern of `stale_pred_full.py`. Label-free, self-check passes (the same code with the witness mask reproduces
   pre2both / pre2full: fe, new, _s bitwise, z to 4.8e-6). NOT run (GPU), not reviewed by anyone else:
   `nn_x/w12/stale/hz2n_review/stale_pred_lt3.py`, files in `nn_x/w12/stale/hz2n_review/cf_lt3/`.
   If hz - ref under it stays within 1 SE of the SVF read, issue 5.1 is closed for the decision.
4. A training view with the rule "counter < 3" (nc extended by L rows) is a five-minute CPU build with the builder's code. Worth a GPU slot only if hznFS or hz2nF separates from its base;
   by my own rule R-1 the recommendation is met for view a and, for view c, only on the reading against the view's own defect. Both readings are reported; the call is the lead's.

## 7. Rule notes
- Declaration before numbers: DECLARATION.txt 07:52:23Z. ADDENDUM R-1 (08:06:42Z) was written AFTER the table that prompted it and says so; its size measurement (n8b) came after it.
- One single foreground `sleep 45` (not a chain). At most two heavy processes, 12 threads, peak RSS 15.9 GB. No process signalled. No GPU process. Writes only in my directory and this file.
- Scope: `build_cf_lt3.py`, `cf_lt3/`, `stale_pred_lt3.py` go beyond the review; they are optional and marked as such.
