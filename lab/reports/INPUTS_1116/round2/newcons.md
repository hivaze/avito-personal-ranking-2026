# Round 2, explorer "new constructions": six label-free inputs built at parity

Written 2026-11-16 05:25 UTC. CPU only (12 threads, one process at a time), no GPU process, no training, nothing submitted, no external service. Every file of mine is in `reports/INPUTS_1116/round2/newcons/`; nothing outside it was written (checked with `find -newer`). Paths are relative to `<repo>`. "cz" is the calibrated score statistic of `reports/INPUTS_1116/residual_rows/condlib.py`; here the null model always holds the pool score, the lens's 83-84 columns of the ranker's own inputs **and nx3** (`nx_rp3_z`, `nx_rp3_rk`).

## 0. Answer

**No construction survives. Nothing here is a new piece.** By the rule written before any array existed (`newcons/PRELIST.txt`, 04:56:50Z), none of the six has the same form at |cz| >= 2.5 with the same sign on both witnesses. The largest whole-feed values are 2.00 on the holdout and 2.80 on the stale-val twin, never for the same form [`newcons/out/gates.txt`].

- **(a) Cold candidates.** Three columns that reach cold rows through fine content cells are alive on their own: alone they order cold rows at a within-feed AUC of 0.60-0.64, against 0.75 for the pool. Given the model's score and inputs they add nothing on the whole feed.
- **Cold rows read separately: positive but small, and above 2.5 on val only.** The three columns read +1.2 to +2.9 on the val twin (+2.3 to +2.9 for N2 and N3), +1.8 to +2.3 on natural val, +1.2 to +2.0 on a third set of holdout feeds and 0.0 to +1.0 on the lag-view holdout. By the word fixed in `newcons/FOLLOWUP.txt` this is "VAL ONLY". Its likelihood on val is at most 0.19 of nx3's, so at nx3's forecast of +0.0003 it is worth about +0.00005 per model.
- **(b) Order and recency.** The forward-only transition count adds nothing. The walk seeded with the newest 8 events is the same information as nx3: +3.7 to +3.8 without nx3 in the null model, +1.1 to +1.7 with it.
- **(c) Audience.** Similarity between the user vector and the user vectors of the candidate's audience adds nothing, at item level or at cell level. The content vector already carries it.

One side observation, not promoted: on warm rows the three cell columns read negative on the holdout (-2.3 to -3.2) and on val (-1.5 to -2.9). On the third set of feeds the read is -1.4 at most, below the bar I had fixed for a replication.

## 1. Table

Sources: `newcons/out/gates.txt` (table at the end, detail above it), `newcons/out/gates.json`. Coverage = share of rows with a non-zero raw value. H = the 15,225 holdout feeds in the lag view (pool = 16 held-out wide students). V = val rows with the stale-val twin of the column and of nx3 (base TO16; base FILE in brackets). Entries are z form / rank form.

| Construction | Coverage: holdout lag / val / val twin / public / private | Parity (C3, C4, C5) | cz H | cz V twin [FILE] | cz on cold rows: H ; V | Verdict |
|---|---|---|---|---|---|---|
| N1 `cw_item`: walk from the user's history items to the candidate's cell | 0.652 / 0.734 / 0.702 / 0.734 / 0.732 | **C3 FAIL** (dZ 0.081 / 0.081 / 0.078; bucket 0.112 / 0.112 / 0.118); C4 ok (PSI <= 0.026); C5 ok | -1.88 / -1.77 | +0.32 / -0.21 [+0.44 / -0.18] | -0.05 / +0.14 ; +1.74 / +1.22 | no information beyond nx3; not at parity |
| N2 `cw_cell`: walk from the history's cells to the candidate's cell | 0.895 / 0.935 / 0.926 / 0.942 / 0.942 | C3 ok, thin (dZ 0.044 / 0.044 / 0.042; bucket 0.092 / 0.095 / 0.093); C4 ok (PSI <= 0.031); C5 ok | -1.53 / -1.16 | +0.06 / -0.47 [+0.47 / -0.11] | +0.02 / +0.74 ; +2.49 / +2.33 | no information beyond nx3 |
| N3 `aud_cell`: user vector against the audience of the candidate's cell | 1.000 / 1.000 / 1.000 / 1.000 / 1.000 | ok (dZ 0.000; PSI <= 0.001) | -1.68 / -1.73 | -0.43 / -0.37 [-0.48 / -0.43] | +0.71 / +0.96 ; +2.87 / +2.65 | no information beyond nx3 |
| N4 `fwd20`: forward-only windowed transition, item level | 0.056 / 0.080 / 0.061 / 0.072 / 0.069 | ok (dZ <= 0.020, bucket <= 0.033; PSI z 0.165 / 0.071 / 0.051, rank 0.168 / 0.085 / 0.057) | -1.25 / -0.38 | -2.80 / -1.36 [-2.78 / -1.40] | 0 by construction | no information beyond nx3 |
| N5 `rp3_l8`: the walk of record seeded with the newest 8 events | 0.025 / 0.034 / 0.024 / 0.029 / 0.026 | ok (dZ 0.006, bucket <= 0.013; PSI <= 0.058) | +1.48 / +0.98 | +1.12 / +0.75 [+1.22 / +0.86] | 0 by construction | no information beyond nx3 (it is nx3's information) |
| N6 `aud_item`: user vector against the audience of the candidate item | 0.520 / 0.587 / 0.503 / 0.539 / 0.527 | **C3 FAIL** (private dZ 0.051; val dZ 0.056, bucket 0.104); C4 ok (PSI <= 0.104); C5 ok | -1.21 / -2.00 | -1.67 / -1.90 [-1.39 / -1.52] | 0 by construction | no information beyond nx3; not at parity |

- **Threshold.** |cz| >= 2.5, same sign, same form, on H and on V. Placebo: 30 within-feed permutations per witness read sd 0.70 (H) and 0.79 (V), largest |cz| 1.57 and 2.08. Both sd are below the 0.8-1.2 band of the PRELIST, so its scaling clause applies: 1.75 on H and 1.98 on V. No form passes at the scaled threshold either. The closest is N6 in rank form, -2.00 / -1.90, a negative sign that exists only with nx3 in the null model (without it: -0.03 / -0.46).
- **The pipeline is the nx3 one.** With nx3 left out of the null model, the nx3 files read +4.98 / +4.32 on H and +6.46 / +4.78 on V (FILE +6.51 / +4.79), the values of `nn_x/w12/nx3/out/nx3_gates.txt`.
- **None of the six is a copy.** R2 of the z column with the null model is 0.23 to 0.67.
- **Cold rows** are rows whose item has no admitted item-level posting in the row's own view: 0.480 of lag-view holdout rows, 0.413 of val, 0.497 of the val twin, 0.461 of public, 0.473 of private. The cold-row form is re-computed among the cold rows of the feed and is 0 elsewhere.

## 2. What was built

### 2.1 The attribute

The raw data hold no microcategory, location, price or seller field (`reports/INPUTS_1116/raw_fields.md` section 1). Attribute-level collaborative columns at 256 to 4096 content clusters are on record and null (`clu_pmi*`, `clu_hcnt*`, ICV; `reports/feat_clu_.md`, `residual_rows/out/all_columns.tsv`). A walk at that resolution would have rebuilt a null family, so I used a finer attribute, fixed before any read:

- **Fine content cell** = the stored 4096-cluster of `data/clu_work/item_cl_4096_*` split by a spherical k-means inside the cluster, k2 = clip(round(train rows / 64), 8, 1024), fitted on train item-table rows only.
- 1,264,539 cells; median 59 train items per cell; mean cosine to the own centroid 0.958 [`newcons/out/cells.log`].
- Every item-table row of every split is assigned through its own split's content vector. 14-16% of history events and 0.003-0.016% of candidates have no vector and no cell.

### 2.2 The six constructions

All six use the nx3 view through `nn_x/w12/nx3/nx3_common.py`, imported unchanged: the same member lists, dense and sparse members, kernel time, own-user skip, cap of 500 postings and degree rule. Full definitions: `newcons/PRELIST.txt`; code: `newcons/nc_common.py` (`kern`).

| # | Name | Definition |
|---|---|---|
| N1 | `cw_item` | Over the latest 500 admitted postings of the candidate's CELL (the candidate's own item left out), each distinct sequence adds the sum of wd(x) / len over its events outside that cell; wd(x) = max 0.95^hpos of ITEM x in the row's history / dense degree. Value = total / (sequences + 10) |
| N2 | `cw_cell` | The same scan, with the history matched through cells: wdc(cell) = max 0.95^hpos over history events in that cell / dense cell degree |
| N3 | `aud_cell` | Over the same sequences: mean cosine between the row's user vector and the sequence owner's user vector (centred by the mean train user vector), shrunk by 5 |
| N4 | `fwd20` | Over the sequences that hold the candidate item: for each occurrence, the history items found 1 to 20 steps OLDER add w / d. This is the directional half of the input `seq_cv_invd20` |
| N5 | `rp3_l8` | Kernel of record `c9.rp3`, unchanged, with weight 0 for every history event beyond the newest 8 |
| N6 | `aud_item` | As N3, over the sequences that hold the candidate ITEM |

Entered forms, fixed in the PRELIST: z = within-feed z (of log1p(K x) for N1, N2, N4, N5; of the raw value for N3, N6), clipped to 5; rank = within-feed mid-rank percentile minus 0.5.

### 2.3 Views and checks

Built for the 15,225 holdout feeds that have the lag view, for val (natural and stale-val twin) and for test (natural). No full-train array. Build time 474 s [`newcons/out/build.log`].

| Check | Result | Source |
|---|---|---|
| My driver against nx3's stored raw arrays (column `rp3_rec`) | max abs diff 0, 0 zero-pattern mismatches on 2,366,505 holdout rows, 1,389,971 val rows (natural and twin) and 2,904,806 test rows | `out/build.log`, lines "driver check" |
| Causality audit, every admitted (row, sequence) pair, item postings | 0 violations of 18.1M (holdout), 17.4M (val), 32.9M (test), 12.9M (twin) | `out/build.log` |
| The same for cell postings | 0 violations of 303M, 249M, 524M, 221M | `out/build.log` |
| Independent numpy re-derivation, 34 feeds, 4,240 rows | largest relative difference 9.9e-15 (8.5e-07 for the two audience columns, which sum float32 products); 0 zero-pattern mismatches | `out/build.log` |
| `fwd20` + backward half against the stored input `seq_cv_invd20`, val | corr 0.974 raw, 0.997 after log1p; non-zero share 0.1132 against 0.1135 | `out/gates.txt` part 0 |

No label array is opened by any build script. The audit counts the five violation kinds of `nx3_common.audit`: dense member at or after the cut, sparse member at or after the own timestamp, a sequence of the own user, any member at or after the own timestamp, the own feed as a member.

## 3. Reading by priority

### 3.1 (a) Cold candidates

**The columns work, and the model already has what they carry.** Mean within-feed AUC of the raw column alone [`newcons/out/followup.txt`, lines F4]:

| Column alone | Holdout lag: all / cold / warm rows | Val twin: all / cold / warm rows |
|---|---|---|
| N1 `cw_item` | 0.591 / 0.597 / 0.580 | 0.602 / 0.607 / 0.593 |
| N2 `cw_cell` | 0.614 / 0.632 / 0.598 | 0.624 / 0.639 / 0.608 |
| N3 `aud_cell` | 0.590 / 0.612 / 0.573 | 0.595 / 0.614 / 0.580 |
| pool score | 0.744 / 0.757 / 0.720 | 0.732 / 0.748 / 0.709 |

Given the pool, its inputs and nx3, the whole-feed cz of the three lies between -1.9 and +0.5 on both witnesses (section 1).

**Cold rows alone.** Because cold rows are the point of this priority, I added a third witness after the gate read, declared in `newcons/FOLLOWUP.txt` (05:14:30Z) before it was built: E = the 12,101 holdout feeds of Sep 7-11 that have no lag view, in a fresh view (the same driver with lag 0). These feeds had not been read. cz of the cold-only form, z / rank [`out/followup.txt`, lines F1]:

| Column, cold rows only | V twin | V natural | E (new feeds, fresh view) | H (lag view) | Likelihood gain per feed on V twin, z form |
|---|---|---|---|---|---|
| N1 `cw_item` | +1.74 / +1.22 | +1.75 / +2.04 | +1.31 / +2.04 | +0.04 / +0.22 | 0.00011 |
| N2 `cw_cell` | +2.49 / +2.33 | +1.93 / +2.00 | +1.43 / +1.77 | +0.07 / +0.77 | 0.00022 |
| N3 `aud_cell` | +2.87 / +2.65 | +2.29 / +2.02 | +1.31 / +1.24 | +0.69 / +0.96 | 0.00029 |

- **Word fixed in advance: "VAL ONLY".** The bar was cz >= +2.5 on V and >= +2.0 on E or H in the same form. Only N3 reaches 2.5 on V, and it reads +1.3 on E and +0.7 on H.
- **The sign is the same in all 24 reads of this table**, so a small real effect is likely; the three columns share cells and sequences, so this is one observation, not three. (Under the gate's own null model, which lacks the fresh-view walk, the smallest holdout value reads -0.05 instead of +0.04.)
- **Size.** nx3 gains 0.00154 nats per feed on V and 0.00044 on H [`nn_x/w12/nx3/out/nx3_gates.txt`]. The best cold-row column gains 0.00029 on V (0.19 of nx3), 0.00003 on E and 0.00001 on H. At nx3's forecast of +0.0003 per model that is about +0.00005.
- **Where it sits on val** (post hoc slices): feeds with M >= 50 read +3.6 to +4.1 for N2 and N3, feeds with M < 50 read -0.8 to -1.4; Sep 24 reads +2.1 to +3.0, Sep 22 +0.8 to +1.0. The holdout holds only M >= 50 feeds, so feed size does not explain why it reads lower.
- **Agreement with the record.** `reports/LIT_1115/CRITIC.md` 2.3 found a different cold-row completion "weak" on val (likelihood ratio 4.4 in z form, 11.4 in rank form); that read was on val only. This adds two holdout reads, both lower than val.

The owner's standing instruction is not to reject on the holdout alone. This case is not rejected on the holdout: it fails the pre-declared whole-feed gate on val itself (N1 +0.3 / -0.2, N2 +0.1 / -0.5, N3 -0.4 / -0.4), and the cold-row read is too small to matter even at its val value.

**Parity of N1.** `cw_item` needs item-level overlap with the history, so its coverage follows the density of the index: 0.65 on lag-view holdout rows against 0.73 on val and test rows, and the gap is the same in every listing-age bucket. It fails C3 by a factor of 1.6.

### 3.2 (b) Order and recency

- **Forward-only transitions (N4).** Without nx3 in the null model the column reads +1.14 / +1.88 (H) and -1.07 / +0.12 (V): nothing. With nx3 it turns negative, -2.80 in z form on V and -1.25 on H, with rank forms of -0.4 and -1.4. The share of the windowed weight that is forward is 0.52 on val, so there is little asymmetry to read.
- **Newest 8 events as seeds (N5).** +3.81 / +3.30 (H) and +3.71 / +3.13 (V) without nx3; +1.48 / +0.98 and +1.12 / +0.75 with it; +1.67 / +1.51 on E. The walk's information does not depend much on which part of the history seeds it. The remainder is positive on all three sets of feeds at 0.00004-0.00005 nats per feed, a tenth of nx3 on the holdout. A sharper decay inside nx3 would be a tuning of nx3 worth about that, not a piece.

### 3.3 (c) Audience

- **Item level (N6).** +0.85 / -0.03 (H), +0.02 / -0.46 (V twin), +0.92 / +0.28 (E) without nx3; natural val +2.24 / +1.17. With nx3 the sign is negative. The column covers 52-59% of rows, three times the walk's coverage, and alone it reads AUC 0.57-0.58, yet it carries nothing the model lacks.
- **Cell level (N3).** Section 3.1.
- **Why [judgement].** The content vector comes from a two-tower model, so its product with the user vector already estimates how close the user is to the item's audience. Measuring the audience directly repeats that input.

### 3.4 Side observation: warm rows read negative

The pre-declared split read, and its re-read with sizes [`out/followup.txt`]:

| Column, warm rows only | H (lag view) | V twin | E (new feeds) | V natural |
|---|---|---|---|---|
| N1 `cw_item` | -2.45 / -2.55 | -1.62 / -1.53 | -0.19 / +0.06 | -0.97 / -0.92 |
| N2 `cw_cell` | -2.27 / -2.29 | -2.46 / -2.56 | +0.12 / -0.51 | -2.01 / -2.70 |
| N3 `aud_cell` | -3.02 / -3.17 | -2.89 / -2.68 | -1.38 / -1.25 | -1.97 / -1.35 |

- **Word fixed in advance: "NOT REPLICATED"** (bar: <= -2.0 on E).
- For N3 the sign is the same on all three independent sets of feeds, and the likelihood of the cold and warm forms together is about a third of nx3's on H and V and a sixth on E.
- My reading [judgement]: the model weights content affinity once for both kinds of row, a little too much where item-level counts exist and a little too little where they do not. That is a re-weighting of what the model reads, not a missing input.
- It was seen in a split that was declared as "reported, not a gate", among 48 such reads, and nothing is promoted on it.

## 4. Looks taken

| Block | cz values of candidate columns | Deciding |
|---|---|---|
| `out/gates.txt`: 4 null models x 6 constructions x (2 given nx3 + 2 without + 2 cold + 2 warm), plus by period and by val day | 240 (24 are zero by construction) | 24 (6 constructions x 2 forms x 2 witnesses) |
| `out/followup.txt`: E whole feed, cold and warm forms on 4 views, slices | 252 | 0 (two words fixed in advance, both negative) |
| Placebo permutations | 120 + 150 | - |

Subset placebos: sd 0.94 to 1.05, largest |cz| 2.51 (120 permutations). Whole-feed placebo on E: sd 1.02, largest 2.57.

## 5. Limits

- The lens is linear in the two forms of a column and reads in sample. An input that matters only through an interaction is seen only where I split by cold and warm rows.
- The cell granularity (1.26M cells) was fixed in advance and not varied. A coarser attribute is on record as null; a finer one approaches item identity, which is nx3.
- No full-train array exists, so the parity gates use the holdout feeds as the train side. C3 for N2 passes with thin margins (bucket 0.092-0.095 against 0.10).
- Witness E is in-period and fresh. It is an independent set of labelled feeds, not a staleness-matched one.
- No reweighting to the private mix was run.
- On val the twin replaces only my column and nx3; the control block holds the stored natural inputs, as in `nx3_gates.py`.

## 6. Recommendation

- **No trained arm, and no change to the nx3 forecast.** The largest effect found here has about a fifth of nx3's likelihood on val and under a tenth on the holdout, and nx3's central forecast is +0.0003 per model.
- If a later bundle is trained for another reason, `aud_cell` (z and rank) is the only column I would let ride along: it is at parity trivially (dense, PSI <= 0.001) and a full-train build is about one hour at 12 threads by the measured rate (13,362 feeds in 52 s). I would not spend students on it.
- Priority (a) is answered in the negative for collaborative completion through content: with this, the 4096-cluster forms and the nearest-neighbour form of 11-15, three resolutions have now been read.

## 7. Files (all under `reports/INPUTS_1116/round2/newcons/`)

| File | Content |
|---|---|
| `PRELIST.txt` | the six constructions, forms, gates and the pass rule, 04:56:50Z |
| `FOLLOWUP.txt` | third witness, subset reads and the two words, 05:14:30Z, after the gate read |
| `nc_cells.py` -> `out/cells.log`, `out/cells/` | fine content cells: `item_cell_*`, `hist_cell_*`, `cand_cell_*`, `cell_base.npy`, `sub_cent.npy` |
| `nc_common.py` | driver on top of `nx3_common.py`: cell postings, kernel `kern`, audit, numpy re-derivation |
| `nc_build.py` -> `out/build.log`, `out/raw/{hold_lag,val_nat,val_twin,test_nat}.npy`, `out/raw/hold_has_row.npy` | raw columns (10 per row: `fwd20, bwd20, aud_item, n_item, cw_item, cw_cell, aud_cell, n_cell, rp3_rec, rp3_l8`); `out/build_smoke.log` is the 300-feed test run |
| `nc_gates.py` -> `out/gates.txt`, `out/gates.json` | parity gates, cz on both witnesses, placebo, table |
| `nc_build_E.py` -> `out/build_E.log`, `out/raw/hold_early_fresh.npy` | third witness, fresh view |
| `nc_followup.py` -> `out/followup.txt` | reads F1-F4 |

Disk: 5.2 GB, of which `out/cells/` 4.1 GB and `out/raw/` 1.1 GB. No process of this task is left.
