# Were ideas closed too strictly? A quantitative scan of the catalogue

Scope: the 312 rows of the catalogue in `docs/07-what-failed.md` (lines 337-731), read-only. Every row is in `catalogue.csv` next to this file (scripts and outputs in `work/`).

**Version scanned.** `docs/07-what-failed.md` was edited while this scan ran (mtime 2026-11-19 17:07 UTC). All line numbers below refer to the version with sha256 `ac87275fa6b8a975638a3df93151c581d481db8c9401b74e8b9258f717de22a1`. Every row sits 2 lines lower than in the version first parsed. A few cells gained seed-paired last-checkpoint reads (rows 380, 382-384, 581, 582). `docs/03-solution.md` and `docs/04-validation.md` were edited too (17:09 and 17:04 UTC); citations to them use the current line numbers.

**Note of 2026-11-20.** The catalogue was edited after this scan. Today's page counts 58 adopted, 22 superseded, 155 no effect, 45 hurt, 10 inconclusive and 22 not run, and its line numbers have moved. The records cited below under `docs/records/` (the decision ledger, the status log and the coverage map) are working records and are not published.

## Bottom line

- **Count.** 214 rows are closed (157 no effect, 45 hurt, 12 inconclusive). Among the 186 that are model or pipeline changes, 156 give a validation number. 55 of those are positive (36%).
  - Without the "hurt" rows: 52 of 112 non-zero reads are positive (46%, 95% CI 37-56%). That is what pure noise would give.
  - The positive share is **not above chance**.
- **z distribution.** 93 closed model rows quote an SE. Of these, 11 have 1 ≤ z < 2 (pure noise would give 12.6) and 2 have z ≥ 2 (noise: 2.1). The left tail is heavy instead: 14 rows at z < -2 against 2.1 expected. The closures look like nulls plus real harms, with no visible excess of moderate positives.
- **Hidden real effects.** A mixture model (null / +0.0003 / harmful) fitted to the 93 reads gives a maximum-likelihood estimate of **0** real +0.0003 effects. The 95% upper bound is **13 of 93 rows** (14%), and 8 rows for +0.0005.
  - The data cannot rule out a handful, because the tests had little power. At each row's own SE, a z ≥ 2 test passes a true +0.0003 with median probability 0.13.
  - Re-tests agree. Ten ideas whose first read was positive (mean +0.00053) were re-read in a stronger form of the same kind. Their mean became **-0.00001 ± 0.00013**.
- **Strongest leads.** Two cooldown changes from the click-label era are probably real gains at member level: R-Drop at the last checkpoint, and auxiliary-loss weight 1.0. Their member-level likelihood ratios for +0.0003 against zero are 189 and 48. Their pool-level reads are flat. Neither was re-tested on students in the cooldown-only form.
- **Process errors.** Two retrieval variants (rows 387, 388) were closed although their 10-epoch rule said PROCEED, and their family reads are not on record.
- **Stale or wrong rows.** At least 38 rows are stale, inconsistent with their source or point to a source that lacks the number (section 6). Most material:
  - Row 424 has its direction reversed.
  - Rows 423, 471, 558 and 580 are in the final recipe but are not marked adopted.
  - Rows 436, 461, 590, 631 and 656 say "never run" or "not run" for things that ran or were declared.

## 1. What was parsed

`catalogue.csv` has one line per catalogue row (312 rows; 55 adopted, 22 superseded, 157 no effect, 45 hurt, 12 inconclusive, 21 not run, matching `docs/07-what-failed.md:5`).

**Annotation scope.** The 214 closed rows were annotated by hand. Test strength and closing rules were added from the sources: four read-only audits followed each row's "Where" column into `experiments/*/README.md`, `lab/` records and `docs/records/decision-ledger.txt`. The adopted, superseded and not-run rows carry only an automatic parse: the first signed number in the cell, and the first "± / holdout" pattern after it.

**Columns:**
- `val_delta`, `val_se`, `val_z`, `holdout_delta`, `holdout_se`: the primary read. This is the first-listed arm of the row, or a later and stronger read of the same idea when the row reports one (for example the 8-seed extension in rows 382 and 384, or the all-data file in row 487). When the row quotes two levels instead of a difference, the delta is computed from them and `val_derived` = 1.
- `best_arm_*`: the most favourable arm quoted in a multi-arm row.
- `test_level`, `seeds`, `epochs`: as far as the row or its source says.
- `closure_reason`: one of rule bar, clear loss, holdout veto, 2-epoch screen, single seed, flawed test, member gain pool loss, public read, replication failed, mechanism/bound, other. `reason_detail` gives the rule and the source line.
- `flag`: marks the seed-luck rows (section 5).
- `row_type`: model (a pipeline change: 186 closed rows), leak (9), method (validation protocol, audits, forensics: 18) or control (1).

**Closure reasons, 186 closed model rows:**
- rule bar 104
- clear loss 31
- flawed test 11 (residual gate, sub-list students, bar against a lucky draw)
- other 9
- holdout veto 7
- public read 6
- member gain, pool loss 5
- mechanism/bound 4
- single seed 3
- 2-epoch screen 3
- replication failed 3

**Test levels of the 93 rows with an SE:** 51 pool or file, 22 member, 7 residual or nested CPU reads, 13 proxy or other.

## 2. Positive share and z among closures

| Set (primary validation read) | n with a number | positive | zero | negative | share positive (excl. zero), 95% CI | two-sided p vs 0.5 |
|---|---|---|---|---|---|---|
| all closed rows | 166 | 58 | 4 | 104 | 0.36 [0.28, 0.44] | 0.0004 |
| closed model changes | 156 | 55 | 3 | 98 | 0.36 [0.28, 0.44] | 0.0006 |
| closed model changes, no effect + inconclusive | 115 | 52 | 3 | 60 | 0.46 [0.37, 0.56] | 0.51 |
| closed model changes, hurt | 41 | 3 | 0 | 38 | 0.07 | 1e-8 |
| closed model changes, not a clear loss | 125 | 54 | 3 | 68 | 0.44 [0.35, 0.54] | 0.24 |
| closed model changes, single-arm rows only | 102 | 41 | 2 | 59 | 0.41 [0.31, 0.51] | 0.09 |
| closed model changes, most favourable arm of each row | 166 | 79 | 3 | 84 | 0.48 [0.41, 0.56] | 0.75 |

If every closure were a null, about half would read positive. Adoption takes a little off the right tail (a null passes z ≥ 2 about 2.3% of the time), so slightly under half is expected. Even with each row's most favourable arm, the share is 0.48. **The positive share is not above chance.** It is below chance when the "hurt" rows are counted, because those are real harms.

z of the 93 closed model rows that quote an SE (`work/analyse.out`):

| z | rows | share | expected under all-null |
|---|---|---|---|
| < -2 | 14 | 0.15 | 2.1 |
| -2 to -1 | 23 | 0.25 | 12.6 |
| -1 to 0 | 16 | 0.17 | 31.7 |
| 0 to 1 | 27 | 0.29 | 31.7 |
| **1 to 2** | **11** | 0.12 | **12.6** |
| ≥ 2 | 2 | 0.02 | 2.1 |

The 13 rows at z ≥ 1, and what became of each (docs/07 line):

| Row | Read | What became of it |
|---|---|---|
| 350 | 2025 winners' recipe: +0.00067 ± 0.00045 against the team's CatBoost, a tree-harness comparison | the member missed its bar of 0.3933 by 0.011 |
| 423 | learning rate 2e-3 at 10 epochs: +0.0008 (z 1.4) / +0.0007 | it is the final setting (`docs/03-solution.md:263`; inconsistency 1) |
| 469 | clusters, tree harness: +0.001 ± 0.0005 (z 2.0) | the source's 8-seed +0.00125 ± 0.00045 falls to -0.00006 ± 0.00054 in another column order (`experiments/feature-families/README.md:50,90`); a harness artefact |
| 488 | trio from the second input search: +0.0004 ± 0.0004 | the lagged holdout read -0.0002 ± 0.0003; nothing trained |
| 519 | popularity-free members: +0.00036 ± 0.00030 | re-tested as row 531; the fresh seeds failed to replicate |
| 528 | A1: +0.00017 ± 0.00014 | public -0.00017 at a paired SE of about 0.00018 |
| 536 | popularity-light members inside the teacher: +0.00042 ± 0.00032 | re-opened by the 11-15 audit; this is the re-test; closed on a stale-validation bar (shortlist 4) |
| 558 | three fine-tune seeds: pool +0.00018 ± 0.00012 | kept in the final recipe (`docs/03-solution.md:445,496`; inconsistency 3) |
| 559 | fine-tune HPO: +0.00055 ± 0.00041 | the second member set read -0.00009 ± 0.00027 |
| 574 | self-distillation: +0.00064 ± 0.00041 | superseded by ensemble distillation, which was adopted (row 575) |
| 587 | teacher-labelled user fold: +0.00069 ± 0.00057 | the 2-seed read at z 2.21 shrank; never re-tested (shortlist 3) |
| 632 | consensus: +0.00022 ± 0.00018 | holdout -0.00050 (z -2.68); coverage 4.5% against a gate of 10% |
| 697 | trees with validation labels: +0.00206 ± 0.00101 | the network analogue re-test read +0.00014 ± 0.00054 (row 564) |

So 3 of the 13 are already in the final pipeline in some form, 6 were re-tested or contradicted by another witness, and 2 are tree-harness reads. Only 2 (rows 536 and 587) stay open on weak evidence.

## 3. How many real effects of +0.0003 or more could be hidden

### Power of the tests that closed them

| Test | null passes | true +0.00025 passes | true +0.0005 passes | Source |
|---|---|---|---|---|
| 8-seed pool rule (both witnesses ≥ 0, pooled z ≥ 2) | 0.25% | 20% (Monte Carlo) | 56% (A/A calibration); 67% (Monte Carlo) | `docs/04-validation.md:158`; `lab/nn_x/w12/moe/why/noise/n4_power.txt:16-18` |
| 4-seed screen of 19 Nov (PROMOTE: val ≥ +0.0003, holdout ≥ 0, p ≤ 0.2; KILL: val < 0) | promote 11%, kill 50% | promote 32%, kill 32% | promote 58%, kill 18% | `lab/nn_x/w12/moe/why/verify_noise/v3_power_actual_rule.txt:4-6` |
| 2-seed screen it replaced | 15% | | 36% | `docs/07-what-failed.md:307` |
| seeds per arm for 80% power at +0.0003 | about 32 | | | `n4_power.txt:27` |
| residual gate on stored predictions | | sees 10-26% of a true effect | | `docs/07-what-failed.md:184` |
| public board, near copies | detectable effect 0.00112 | | | `docs/04-validation.md:144` |

At each row's own validation SE (median 0.00035, 10th to 90th percentile 0.00017-0.00063), a z ≥ 2 test passes a true +0.0003 with probability:
- median 0.13 (interquartile 0.08-0.21);
- at least 0.5 for only 8 of 93 rows;
- for +0.0005: median 0.28, at least 0.8 for 11 rows.

A bare "did not pass" therefore carries a median likelihood ratio of 0.89 (real +0.0003 against null) (`work/mixture.out`). One closure is almost no evidence against a +0.0003 effect. Only the distribution of the reads constrains how many there are.

### Mixture estimate (`work/mixture.py`)

Assumptions:
1. Each closed read d_i ~ N(δ_i, s_i²) with the quoted SE. The SE is taken as honest: rms z 1.04 over 14 same-recipe pairs (`docs/04-validation.md:133`), though randomization nulls ran 0.87 to 1.43 (`docs/04-validation.md:162`).
2. δ_i is 0 (null), +Δ (real), or negative (harmful, half-normal with a free scale).
3. Closed rows are conditioned on not passing, with the pass region approximated as z ≥ 2.
4. Rows are independent and the effect is on the scale of the read that closed them (member, pool or file).

| Model | rows | π(+Δ) MLE | 95% profile CI | count, 95% CI |
|---|---|---|---|---|
| three groups, Δ = +0.0003 | 93 | 0.00 | 0.00-0.14 | 0-13 |
| three groups, Δ = +0.0005 | 93 | 0.00 | 0.00-0.09 | 0-8 |
| two groups (null vs +0.0003), rows not closed as a clear loss | 78 | 0.00 | 0.00-0.18 | 0-14 |
| two groups (null vs +0.0005), same rows | 78 | 0.00 | 0.00-0.12 | 0-10 |

The fitted harmful share is 0.14 with scale 0.0021.

Sensitivity (`work/sens.py`, `work/sens.json`), Δ = +0.0003, three groups:

| Variant | rows | count, 95% CI |
|---|---|---|
| SE × 1.2 | 93 | 0-9 |
| SE × 1.4 | 93 | 0-9 |
| most favourable arm of every row (biased upward by best-of-k selection) | 108 | 0-41 (MLE 18) |
| most favourable arm, SE × 1.2 | 108 | 0-32 |
| most favourable arm, SE × 1.4 | 108 | 0-25 |

In the two-group model 23 of 78 reads favour +0.0003 over null (likelihood ratio above 1), none by more than 3 to 1, and 27 favour null by more than 3 to 1. The mean z of those 78 rows is -0.16 ± 0.11.

**Reading.**
- The best estimate is that no closure hides a real +0.0003. The data allow up to about 13 of the 93 rows (14%).
- Extrapolated at the same rate to the 93 closed model rows without an SE, the bound is about 25 of 186.
- The bound reaches 40 only if every row is read at its most favourable arm, which selection inflates.
- This is weak evidence either way. A real +0.0003 moves a read by 0.9 z at the median SE, so a 14% share could hide in 93 noisy reads.

### Re-test record (`work/retest.py`)

Ten ideas had a positive first read and a later stronger read of the same kind on validation:

| Row | Idea | First read | Later read |
|---|---|---|---|
| 387 | unique retrieval slots | 2-epoch +0.00117 | 10 epochs +0.00036 ± 0.00068 |
| 531 | popularity-free second view | +0.00036 ± 0.00019 | fresh seeds -0.00024 ± 0.00018 |
| 458 | per-feed weight M^0.5 | +0.00036 ± 0.00034 | all-data file -0.00002 ± 0.00027 |
| 534 | stale-view cooldown | pool +0.00024 ± 0.00038 | file -0.00047 ± 0.00035 |
| 380 | width 224 | +0.00026 ± 0.00058 | files +0.00005 to +0.00017 |
| 524 | feed-size rule | in-sample +0.00051 ± 0.00021 | nested +0.00002 ± 0.00027 |
| 507 | forward echo of f5 | +0.00038 ± 0.00024 | -0.00015 ± 0.00038 |
| 559 | fine-tune HPO | +0.00055 ± 0.00041 | -0.00009 |
| 562 | random-order feeds in the fine-tune | +0.0005 | mean -0.00017 |
| 588 | pseudo-feeds | 2-seed member screen +0.00092 ± 0.00060 | 8-seed pool +0.00054 ± 0.00047 |

- Mean first read +0.00053, mean later read **-0.00001 ± 0.00013**.
- 2 of the 10 kept at least +0.0003, and 1 of 10 kept half its first value.
- With six cross-regime re-tests added (for example self-distillation to ensemble distillation, width 192 from click labels to students), the mean later read is +0.00016. The two regime changes that grew are already adopted.

A weak positive first read here kept about none of its value on average. That fits a mostly-null set of closures.

## 4. Shortlist: closures most likely to hide a real gain

P(real) is the posterior probability of a true +0.0003 on the stated read, under priors of 0.05 and 0.14. The second is the upper bound of section 3. The likelihood ratios are in `work/short.py`.

**1. R-Drop and SAM regularisers in the cooldown (docs/07:428)**
- Read: as catalogued, members at the selected checkpoint: R-Drop -0.00012 ± 0.00032 / +0.00059, SAM +0.00004 / +0.00041. At the last checkpoint, which is the checkpoint all-data members use: R-Drop +0.00057 ± 0.00030 (z 1.89) / +0.00052 ± 0.00017 (z 3.12); SAM +0.00039 / +0.00038, stale +0.00065 (z 2.04). 4-member pools at the selected checkpoint: R-Drop -0.00024 ± 0.00046 / +0.00036 ± 0.00025, SAM +0.00033 ± 0.00045 / +0.00025 ± 0.00023. Strength: 4 roots, click-label cooldown re-runs (`experiments/training-waves/results/wave7_readout.txt:25-38,73-79`).
- Why closed: the selected-checkpoint rule (val > +2 SE).
- Re-tested? A fresh-root replication was re-opened (prior 0.02) and never run (`docs/records/coverage-map.md:241`). Never run on students.
- Still addable? Cheap: one cooldown re-run is about 0.25 GPU-h per member. But members correlate 0.990 with their base, so a file would be a near copy, below public's detectable 0.00112.
- P(real) for a member gain: 0.91 / 0.97. For a pool gain: 0.08 / 0.20 (R-Drop), 0.11 / 0.27 (SAM).

**2. Auxiliary-loss weight 1.0 in the cooldown (docs/07:418)**
- Read: the catalogue gives the 4-root +0.00032 ± 0.00020. The 7-root read is +0.00038 ± 0.00017 (z 2.2) / +0.00019 ± 0.00009 (z 2.0), with the 7-member pool holdout -0.00023 (z -1.5) (`docs/records/experiment-registry.md:205`).
- Why closed: member gain, pool loss.
- Re-tested? Only from scratch for students (row 457): -0.00023 ± 0.00055 / +0.00028 ± 0.00031, pooled z +0.34, 4 seeds. The cooldown-only form under distillation was never run; the student harness had aux 0 only (`docs/records/decision-ledger.txt:180`).
- Still addable? A cooldown re-run. Expect the member-to-pool loss seen at weight 0.8 of the teacher.
- P(real): member 0.72 / 0.89; student pool 0.05 / 0.14.

**3. Teacher-labelled feeds of a held-out user fold (docs/07:587)**
- Read: 4-pool +0.00069 ± 0.00057 / -0.00017 ± 0.00032; members +0.00103 / +0.00052. The 2-seed read had been +0.00158 ± 0.00071 (z 2.21). 4 seeds, 10 epochs.
- Why closed: rule bar. The 4-seed rule needed val ≥ +2 SE and read +1.2 SE, so "information only". B − C on the holdout was -0.00082 (z -2.15), and the extra teacher weight is a confound (`decision-ledger.txt:301-310`).
- Re-tested? Never. Transductive distillation (B1) was declared on 18 Nov (`decision-ledger.txt:1381-1382`) and not run. It is the main remaining option in `docs/08-outside-ideas.md:130-134`.
- Still addable? It needs new students (a screen of about 16 GPU-h). Members trained with more teacher weight moved up while their pool moved down (rows 577, 579).
- P(real): 0.03 / 0.09.

**4. Popularity-light members inside the teacher (docs/07:536)**
- Read: 16-pool val +0.00042 ± 0.00032 (z 1.3), stale +0.00008 ± 0.00030, private-mix about +2 SE; with the stale view -0.00035. 16 cooldown re-runs.
- Why closed: re-opened by the holdout-veto audit, then failed the rule's stale-validation bar (≥ +2 SE; read z 0.3) (`decision-ledger.txt:316-318,498-504`).
- Re-tested? This row is the re-test.
- Still addable? Overlaps the 0.2 popularity-light admixture already in both picks. The stale view kept only 43% of its effect on top of it (`docs/07-what-failed.md:280`).
- P(real): 0.10 / 0.26.

**5. Unique retrieval slots with per-type counts (docs/07:387)**
- Read: 10-epoch pair +0.00036 ± 0.00068 / +0.00018 ± 0.00041; cold positives +0.0095 (z 7.1), warm -0.0034 (z -3.7) (`experiments/rounds-5-7/results/round6_retrieval_slots_10epoch_pair.txt:7-15`). The 2-epoch screen had been z 2.48.
- Why closed: catalogued as no effect, but its 10-epoch rule said PROCEED. The 6-seed family ran and its final read is not on record (`docs/records/experiment-registry.md:83-84,95`).
- Re-tested? The student arm kwa4u was not run, on a reviewer's reading (`decision-ledger.txt:682`).
- Still addable? A cold-for-warm trade. Private has more cold candidates than validation (glossary cold share 0.477 against 0.348, `docs/07-what-failed.md:711`), so its private-mix read decides. That read is not in the row.
- P(real): 0.06 / 0.16.

**6. Set-attention variant and category retrieval (docs/07:388)**
- Read: set variant 10-epoch +0.00017 / -0.00001 (2-epoch holdout +0.00090, z 2.7); category retrieval +0.00046 ± 0.00049 / 0.0, 2 seeds at 2 epochs.
- Why closed: set variant PROCEED with no family read on record. Category retrieval failed the 2-epoch rule (z 0.9) (`docs/records/status-log.md:119,179`).
- Re-tested? Category-retrieval all-data students were queued as a try file on 17 Nov with no read (`decision-ledger.txt:1371`).
- Still addable? It needs 16 all-data students (about 60-70 GPU-h).
- P(real): 0.07 / 0.19.

**7. Mixed teacher (docs/07:580)**
- Read: seed-paired -0.00030 ± 0.00057 / +0.00013 ± 0.00032. Against the 70-pool mean: +0.00027 ± 0.00048 at the selected checkpoint, +0.00095 ± 0.00046 at the last (`lab/nn_x/w12/moe/logs/fair_all_lead.txt:3`).
- Why closed: rule bar, mean ≥ +0.0004 (`decision-ledger.txt:41-48`).
- Re-tested? No. It is uncertain, not wrongly closed: another teacher does not share the reference seeds' luck, so the seed-paired read may be up to 0.0006 too low (`docs/04-validation.md:339-349`).
- Still addable? Only fresh seeds settle it. The 25-member teacher's students are already half of the final full-view students.
- P(real): 0.03 / 0.09, or 0.05 / 0.15 after the full +0.0006 correction.

**8. Top-20 Plackett-Luce target for the auxiliary loss (docs/07:446)**
- Read: +0.00059 ± 0.00045 (z 1.31) / +0.00021; 2 seeds, 2 epochs.
- Why closed: rule bar (val z > 2). No 10-epoch pair, and a CPU stale replay found the gain gone under private-like staleness (`lab/reports/sol_gpu.md:8-17`).
- Re-tested? Never.
- Still addable? It would need a student run.
- P(real): 0.09 / 0.24.

**9. Width 224 students (docs/07:380)**
- Read: 4-pool +0.00026 ± 0.00058 / +0.00016 ± 0.00032; last checkpoint, paired by seed, +0.00031 ± 0.00057 / +0.00021. All-data files +0.00005 to +0.00017, for example FDXU32P20:w224 +0.00017 ± 0.00025 (`decision-ledger.txt:1425`).
- Why closed: 4-seed rule, pooled z +0.68.
- Re-tested? Partly: all-data files were built on 18 Nov and never scored.
- Still addable? The files exist, so a public read costs no GPU. It would be a near copy, below public's detectable effect.
- P(real): 0.06 / 0.17.

**10. Fine-tune on test-like stale inputs (docs/07:537)**
- Read: 8-member pool +0.00022 ± 0.00026, stale +0.00054 ± 0.00028 (z 1.93); the 70-subset average is +0.00017 ± 0.00014, and private-mix for the 8-pool -0.00006 ± 0.00044. One fine-tune seed per member.
- Why closed: missed the stale bar (≥ +2 SE) by 0.07 SE. The reviewer called it "one high draw" (`lab/reports/STALE_1115/staleft.md:175-182`; `staleft_review.md:60-64`).
- Re-tested? The 3-seed run was not done.
- Still addable? Fine-tune only, so cheap, but the fine-tuned half is 82% common mode (row 566).
- P(real): 0.07 / 0.18.

Left off the list:
- Clusters (row 469): column-order artefact.
- Items 1 and 2 at pool level: their member-level likelihood ratios are high, but the scan asks about pool gains, and pools of one recipe saturate. Eight students are within 0.00024 of the asymptote, and 16 against 8 read -0.00007 ± 0.00034 (`docs/03-solution.md:426`; `experiments/pool-size-and-growth/README.md:5`).
- The residual-gate closures (about 300 feature rows; rows 479-485, 670, 673, 649). Their test is known to be blind (`docs/07-what-failed.md:184`). The conditional score test of 16 November re-screened the 284 stored columns and found one survivor, which failed as a trained input (row 487). Whether the remaining residual-closed constructions were among those 284 columns was not checked.

None of the ten could be confirmed on the public board. A file change below about +0.0011 is invisible there (`docs/04-validation.md:144`).

## 5. Rows read against a lucky seed-paired reference (uncertain, not wrongly closed)

Student screens from 14 to 19 November paired arm seeds 951-954 with the reference's seeds 951-954. That reference pool reads 0.40134, 3rd of 70 and 0.00057 above the mean (`docs/04-validation.md:341`). Its luck is shared by same-teacher arms:

| Recipe | val, seeds 951-954 minus 955-958 (`seedluck.txt` line) |
|---|---|
| reference | +0.00113 (line 1) |
| arm A | +0.00082 (line 5) |
| arm C | +0.00085 (line 7) |
| the 25-member-teacher recipe | -0.00051 (line 3) |

Source: `lab/nn_x/w12/moe/logs/seedluck.txt`. So:

- **Same-teacher arms** (rows 380, 381, 382, 384, 457 and the others in `fair_all_lead.txt`): the seed-paired read is the better one. The 70-subset-mean re-reads overshoot by about 0.0006; the 8-seed extension took arm A from +0.00070 to +0.00006. They are not evidence of a gain, and none is used as one above.
- **Arms with another teacher**: the seed-paired read may be biased against the arm by up to about 0.0006. These are flagged uncertain in `catalogue.csv`:
  - row 580, mixed teacher and 25-member teacher;
  - row 581, students of students: against the 70-pool mean -0.00013 and -0.00023;
  - row 582, cross-fitted teacher: against the mean -0.00015 ± 0.00047 / -0.00059 ± 0.00026 at weight 0.5, so the holdout stays negative.
- The current catalogue already notes this for rows 581 and 582. Row 580 does not.

## 6. Inconsistent or stale rows

Line numbers are those of the scanned version of `docs/07-what-failed.md`. Each entry was checked against the current text; sources were read by the four audits and spot-checked.

### Outcome stale or wrong

| # | docs/07 line | Current text | What the source says | Kind |
|---|---|---|---|---|
| 1 | 423 | "Learning rate 1e-3 or 2e-3 \| inconclusive \| … +0.0008 (z 1.4) / +0.0007" | Final recipe uses 2e-3 (`docs/03-solution.md:263-265`). At 10 epochs the holdout is +0.00067, run-SE 0.00023, z 2.9 (`lab/reports/r6_recap.md:174-179`). | outcome (adopted); holdout z omitted |
| 2 | 424 | "Muon also on the input projections \| hurt \| validation -0.00101" | The arm moved the inputs off Muon: "5 input projections on AdamW … not Muon" (`lab/reports/R7_ARCH_HPARAM.md:34`). Verdict "keep Muon on all 2-D matrices" (:138). Also "moving the input projections from Muon to AdamW cost -0.00101" (`experiments/optimiser-and-modern-tricks/README.md:78`). | direction reversed |
| 3 | 558 | "Three fine-tune seeds per fold \| no effect \| … kept as the default" | In the final recipe: "3 seeds per fold" (`docs/03-solution.md:445`), "three were kept" (:496). Adopted per the definition at `docs/07:326`. | outcome |
| 4 | 471 | "Candidate-competition inputs \| inconclusive" | The inputs are in the final 34: `ctx_hn_*` (6), `ctx_hs_*` (4) (`docs/03-solution.md:71`). Only the removal test was inconclusive (`docs/records/coverage-map.md:191`). | outcome |
| 5 | 580 | "Teacher form: 25 members … \| no effect" | Width-192 students of the 25-member teacher were pooled into W8, "two teachers, post hoc \| adopted" (`experiments/distillation/README.md:52`). Both picks use "seeds 951-958 for each teacher" (`docs/03-solution.md:528`). The mixed-teacher arm lacks the seed-luck caveat that rows 581-582 now carry. | outcome; caveat missing |
| 6 | 469 | "clusters +0.001 ± 0.0005, later +0.0007 ± 0.0007" \| no effect | Clusters: 8 seeds +0.00125 ± 0.00045, KEEP, "never a ranker input". The +0.0007 ± 0.0007 is the whole 122-feature family, not a later read. Another column order reads -0.00006 ± 0.00054. (`lab/reports/feat_clu_.md:11,17`; `experiments/feature-families/README.md:50,90,96`) | number; outcome ("kept, never tested") |
| 7 | 457 | weight 0 "-0.00135 / -0.00128" under "no effect" | "pooled z -4.38 … CLOSED on both witnesses" (`decision-ledger.txt:333`). This is the removal of the adopted auxiliary loss. | outcome (hurt) |
| 8 | 446 | temperature 30 under "no effect" | "WORSE", holdout z -3.86 (`lab/reports/sol_gpu.md:9`) | outcome (hurt) |
| 9 | 419, 432 | Null arms under "hurt": cooldown weight decay 0.1 (-0.00055) and equal history length (+0.00007) | -0.00055 ± 0.00053, z -1.04, holdout +0.00029 (`experiments/rounds-5-7/results/round6_cooldown_stage1_2pct_and_weight_decay.txt:19,24`). hcap +0.00007 / +0.00002, "Null" (`docs/records/status-log.md:416`). | outcome (bundling) |
| 10 | 600, 604 | "hurt" at public -0.00035 and -0.00039 | Paired public SE is 0.00035-0.00062 (`experiments/hard-label-pools/README.md:45`), so about 1 SE. Rows 605 (-0.00038) and 584 (-0.00041, 0.95 SE) are "no effect". | outcome (labelling) |
| 11 | 387, 388 | "no effect" | The 10-epoch rule said "PROCEED to 6-seed family" (`round6_retrieval_slots_10epoch_pair.txt:15`). The family's final reads are not on record (`experiment-registry.md:83-85,95`). | closed without a final read |
| 12 | 663 | "Eight ideas … \| no effect \| … 0 of 4 advanced" | Three of the four screened arms have their own "hurt" rows (392, 393, 406) | double count |
| 13 | 685 | "Scan of 211 techniques … \| no effect" | Classified by prior, not run: "82 WEAKLY_CLOSED … 19 rejected by a skeptic without a run" (`lab/reports/EXTERNAL_SCAN_1113.md:20-21`). The page defines that as "not run" (`docs/07:326`). | outcome (definition) |

### "Never" or "not run" when it ran or was declared

| # | docs/07 line | Current text | What the source says |
|---|---|---|---|
| 14 | 461 | "never trained: down-weighting negatives below the last click … within-category auxiliary loss" | Both trained on 17 Nov. casc5 "ignores negatives > 5 positions below the last positive" (`decision-ledger.txt:1281`; row 460). wcl05/wcl10 is "more weight on the order inside categories" (`decision-ledger.txt:1223`; row 458). The within-category prior was P ~0.07, not ~0.02 (`lab/reports/INSIGHTS_1112.md:170`). |
| 15 | 436 | "cooldown fraction and floor", "learning rate above 2e-3 outside the students' cooldown" never varied | A 40% cooldown (cf4) ran (`docs/records/status-log.md:202,314`; row 416). So did lr 4e-3 at 4× batch (bs4lr4, `status-log.md:665,672`; row 410). The records say "NEVER (in a full run)" (`coverage-map.md:27`; `docs/03-solution.md:629`). |
| 16 | 590 | "distillation on test feeds; a teacher weight per feed … never declared" | B1 "transductive distillation on later unlabelled feeds … rule 8-pool val >= +2 SE" was declared and not run (`decision-ledger.txt:1381-1382`). The U20 rule names "val/test feeds as a transfer set" (:130). A per-feed teacher-gap weighting ran (row 459; :1280,1300). |
| 17 | 631 | "Nearest-neighbour lookup of other users' labels … \| not run" | The reopen form (two-stage key, as-of bank) ran as P4, pre-declared 2026-11-13T05:15:18Z (`lab/nn_x/pretrain2/P4/PREDECLARED.txt:2`); it is catalogued at row 646 |
| 18 | 656 | "Feasibility study of small language models … \| not run" | "conditional GO for Stage A only" (`lab/reports/LLM_FEASIBILITY.md:18`). Stage A ran (rows 657-659). |
| 19 | records | coverage-map rows | `docs/records/coverage-map.md:51` says history-event dropout is "NEVER" (kwh30 ran, row 429). :246 lists post-last-click down-weighting as UNTESTED (ran as casc5). :317 says m05 was "still never run" (:364 says run and closed). :362 says MIXFT is "RUNNING" (closed at `decision-ledger.txt:825-828`). |

The two examples named in the brief are now clean. `experiments/external-solutions/README.md:50`, which said mixture of experts was "neither tried", now reads "graph networks | … not tried". No document calls the cross-fitted teacher untested; it is row 582 with its reads.

### Numbers or witnesses

| # | docs/07 line | Current text | What the source says |
|---|---|---|---|
| 20 | 429 | "10 epochs: -0.0008" | An inference replay on trained members, not a training run (`lab/reports/TRICKS_SWEEP_1113.md:261,475`) |
| 21 | 418 | "1.0 in the cooldown +0.00032 ± 0.00020 (z 1.6) … 7-member pool holdout -0.00023" | 7-root member read: val +0.00038 ± 0.00017 (z 2.2), holdout +0.00019 ± 0.00009 (z 2.0) (`experiment-registry.md:205`). The cell pairs a 4-run member read with a 7-run pool read. |
| 22 | 428 | Selected-checkpoint reads only | Last checkpoint: R-Drop +0.00057 ± 0.00030 / +0.00052 ± 0.00017; SAM stale +0.00065 (z 2.04) (`wave7_readout.txt:35-36,73-79`) |
| 23 | 476 | "+0.00027 ± 0.00028 / -0.00008 ± 0.00020" (reads as val / holdout) | Both are validation residuals (`lab/reports/featnew_ud.md:10-11`). The val / holdout gate reads -0.00056 ± 0.00069 / +0.00034 ± 0.00042 (`experiments/feature-families/README.md:76`). |
| 24 | 585 | "-0.00004 / -0.00007" | Two arms on validation, kd05 ×4 and kd08 ×4, not val / holdout (`decision-ledger.txt:18-19`) |
| 25 | 531 | "first draw +0.00036 ± 0.00019 … replication -0.00024 ± 0.00018" | The first number is against control FD16P20w8, the replication against FD16P20. Against FD16P20 the first draw reads +0.00015 ± 0.00020 (`decision-ledger.txt:651,725`). |
| 26 | 537 | "0.00000 ± 0.00022 on the private-mix view" for the 8-member pool | That is the 70-subset average. The 8-member pool reads -0.00006 ± 0.00044 (`lab/reports/STALE_1115/staleft_review.md:60,64`). |
| 27 | 507 | "nested +0.00038 ± 0.00024 (z 1.6)", Where = p09_f5echo.py | That number is the G3 read (`coverage-map.md:224`). The cited script logs +0.00022 ± 0.00027 (`experiments/external-solutions/README.md:66`). |
| 28 | 618 | "+0.00006 ± 0.00029 on validation" | The decisive read was the pre-declared fresh confirmation, -0.00017 ± 0.00031 (`docs/records/status-log.md:444`) |
| 29 | 662 | "four other rules -0.00034 to -0.00070" | Three rules (`lab/nn_x/llm_review_1117/RESULT.txt:36-38`; `experiments/llm-rankers/README.md:97`) |
| 30 | 714 | "its level overshoots about 5×" | Differences overshoot about 5×; the level by +0.0045 (`docs/records/status-log.md:694`) |
| 31 | 731 | "second study on 15 same-recipe seed pairs: rms z 0.83 …, 0.89" | Source is `lab/reports/REVIEW_SYNTHESIS.md:175`, not `lab/nn_x/verify`. `docs/04-validation.md:133` gives 14 pairs, rms z 1.04 (`lab/reports/review_metric.md:13`). |
| 32 | 612 | "the file law predicted +0.000055 from 16 to 32" | `experiments/pool-size-and-growth/README.md:5` says +0.00007 |
| 33 | 381 | "doubled similarity bias … pooled z -1.58" | kwsb2 = `--sim_bias 2`, the "rich bias" of row 374 (`lab/nn/model.py:226`; `docs/glossary.md:139`) |
| 34 | 468 (adopted) | "as ranker inputs +0.0014 / +0.0003" | "(0.0000) … holdout (+0.0014)" (`lab/reports/nn_report.md:266`): witnesses swapped |
| 35 | 644 | "gate … failed: zero-shot 0.359 alone, nested residual … at or below 0" | The zero-shot leg passed (0.3625 ≥ 0.36). The residual and AUC legs failed (0.316 against 0.52) (`experiments/pretraining/results/encoder_as_init_stage0_gate.txt:59-61`). |
| 36 | 682, 683 | Where = e02 for row 682 and e05 for row 683 | Swapped. Row 683's numbers are in e02 (`lab/nn_x/audit/external/out/e02.log:26,28`), and its out-of-fold +0.00002 is category × day, not the same oracle. Row 682's joint -0.00034 is in e05 (`out/e05.log:7`). |
| 37 | 649 | Five processing items; holdout -0.00004 | The pre-declared items differ (`emb_usability_predeclared.txt`). With the validation weight carried over, the holdout reads -0.00017 ± 0.00012 (`emb_usability_holdout.log:4`). |

### The "Where" column names a folder that lacks the number (38th group)

Agreement of number and source was otherwise good. The audits found the cited `experiments/*/README.md` has no passage for these rows; the number lives in the ledger, `docs/records/` or `lab/reports/`:

| Rows | Source holding the number |
|---|---|
| 347 | `experiments/fine-tune/README.md:14-18` |
| 363 | `rounds-5-7/results/review_loss_zoo_report.txt:18` |
| 366 | `status-log.md:610` |
| 386 | `experiment-registry.md:181,184` |
| 408 | `rounds-5-7/README.md:41,67` |
| 416 | `status-log.md:313-317` |
| 560-565 | `nn_report.md`, `coverage-map.md:97,148`, `ft_redesign.md:14`, ledger:1199 |
| 576, 577, 583, 586 | `experiments/distillation/README.md:46-49,63,114`; GRU students only in `experiment-registry.md:263` |
| 585 | `decision-ledger.txt:18-19` |
| 587 | `experiment-registry.md:307`, `coverage-map.md:310`, ledger:301-341 (no README mentions U20) |
| 614 | `decision-ledger.txt:778,835` |
| 618 | `status-log.md:444,733` |
| 627 | `lab/reports/RECHECK_1114/winner_model.md:35,43,52` |
| 629, 632, 684 | `lab/nn_x/insights/ccb_1113/RESULT.md`, `enrich_1113/` (the error-analysis README has no CCB passage) |
| 643 | `lab/reports/pretrain_report.md:114` |
| 695 | `status-log.md:761` |

Also, `experiments/rounds-5-7/README.md:26` describes round 6's test-time arms as "re-shuffled candidates, sub-lists or history lengths", but they were MC dropout, history-event dropout and crops (`lab/reports/r6_tta.md:11-13`).

## 7. Files

All paths are under `lab/nn_x/w12/closures_scan/` (written 2026-11-19 by a read-only scan; scripts were run from a scratch copy).

- `catalogue.csv`: all 312 rows with the columns of section 1 (closed rows annotated by hand; other rows auto-parsed).
- `work/ann.py`: the hand annotations, with the source lines behind each update.
- `work/analyse.py` → `work/analyse.out`: counts and z bins.
- `work/mixture.py` → `work/mixture.out`, `work/mixture.json`: mixture models and power.
- `work/sens.py` → `work/sens.json`: sensitivity.
- `work/retest.py` → `work/retest.json`: re-test record.
- `work/short.py` → `work/short.json`: shortlist likelihood ratios.
- `work/rows_v1.json`, `work/linemap_v1_to_v2.json`: the first parse and its +2 line map to the scanned version.
