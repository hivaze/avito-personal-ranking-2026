# famS_review: adversarial review of family S (CatBoost students on teacher targets + raw embeddings)

Written 2026-11-15 07:00Z. Reviewed: `reports/CATBOOST_1115/famS.md` and the structured result of the famS study (three runs of one task).
Marks: **[M]** measured by me, **[D]** read in a project file or transcript, **[I]** inferred. "a / b" = val / holdout; "±" = 1 SE, paired per feed (val 14,423 feeds, holdout 30,000).
Reviewer's code and logs: `nn_x/w12/cb/famS_review/` (section 8). CPU only, 4 threads, one 20-second diagnostic fit, nothing else trained.

## 0. Verdict

**CONFIRMED. I could not break it.** Every number the verdict rests on was re-derived from the saved prediction files with my own NDCG and conditional-logit code, on witnesses and a W16 pool that I rebuilt from raw files. No leak, no selection on a witness, no changed source or data file, no GPU use.

1. **Nothing is PROMISING, nothing passes the member rule** [M]. All ten models and the declared 3-seed bundle read NATS +0.00030 to +0.00074 val / +0.00023 to +0.00048 holdout against the bar 0.0016 on both. Declared bundle: +0.00056 ± 0.00026 / +0.00040 ± 0.00015; member rule -0.00026 ± 0.00037 / -0.00000 ± 0.00020, pooled z -0.50 (study: -0.00027, z -0.52; float32 against float64 arithmetic).
2. **The negative result cannot be turned by any re-combination of the family's files** [M]. With free weights for all ten models at once (an in-sample upper bound) the family reads +0.00121 / +0.00062; the widest member-equal bundles reach val 0.38509 (all ten), 0.38572 (the three highest), 0.38666 (with the LightGBM plateau student): all at least 0.0033 below the stage-2 gate.
3. **Stage-2 gate not met**: best declared level 0.38461 / 0.30797, short by 0.0054 [M].
4. **Student = subset of the hard-label tree** [M]: given [W16, hard-label tree] the student bundle adds +0.00006 ± 0.00009 / +0.00001 ± 0.00002; the hard-label tree adds +0.00068 ± 0.00029 / +0.00012 ± 0.00008 on top of the student.
5. Two corrections of wording, no number changes (section 6): "0.3846, reached four ways" is a level at the CPU caps, not a measured ceiling (two of the four were still rising, z +3.2 and +3.5); the holdout range of the four is 0.3076 to 0.3086, not 0.3080 to 0.3086.

## 1. Claim by claim

Mine = own code (`rv_own.py`: NDCG by counting with pessimistic ties, conditional logit by BFGS with an analytic gradient, per-feed z by segment sums). `nats.py` was run only afterwards, as a cross-check on `ts_lr020_s0` and on the bundle: it prints the same lines (`logs/nats_crosscheck.log`).

| claim | study's number | my number | agree |
|---|---|---|---|
| W16 train-only pool level | 0.40144 / 0.32460 | 0.40144 / 0.32460 (rebuilt from the 16 member files; equals the infra cache to 6e-7) | yes |
| holdout = 30,000 feeds, none in `xm` | stated | rebuilt from raw train files: M, offsets and labels equal `nn_x/w5/preds/holdout` exactly; overlap with `xm` 0 | yes |
| reference `ts_lr020_s0` | 0.38150 / 0.30462; rho 0.847 / 0.847; NATS +0.00117 ± 0.00038 / +0.00051 ± 0.00017; member -0.00006 ± 0.00039, -0.00008 ± 0.00022, z -0.36 | identical | yes |
| S1 `s1_rmse_d8_lr010_s0` | 0.38453 / 0.30762; rho 0.898; NATS +0.00057 ± 0.00027 / +0.00036 ± 0.00014; member -0.00049 ± 0.00037, -0.00003 ± 0.00020, z -1.05 | identical | yes |
| S1f seed 0 | 0.38404 / 0.30737; rho 0.897; NATS +0.00062 ± 0.00028 / +0.00036 ± 0.00014; member -0.00031, +0.00017, z +0.02 | identical | yes |
| S1f seed 1 (cut at 886 trees) | 0.38326 / 0.30696; NATS +0.00045 ± 0.00024 / +0.00040 ± 0.00015; z +0.03 | identical | yes |
| S1f seed 2 | 0.38385 / 0.30765; NATS +0.00054 ± 0.00026 / +0.00041 ± 0.00015; z -0.28 | identical | yes |
| **declared bundle (3 seeds of S1f)** | 0.38461 / 0.30797; rho 0.900; NATS +0.00056 ± 0.00026 / +0.00040 ± 0.00015; member -0.00027 ± 0.00037, -0.00000 ± 0.00020, z -0.52 | 0.38461 / 0.30797; rho 0.900 / 0.900; NATS +0.00056 ± 0.00026 / +0.00040 ± 0.00015; member -0.00026 ± 0.00037, -0.00000 ± 0.00020, z -0.50 | yes |
| S1f plateau run `..._s1long` | 0.38451 / 0.30792; rho 0.902; NATS +0.00049 ± 0.00025 / +0.00043 ± 0.00015; z -0.29 | identical | yes |
| S3 (PCA embedding block) | 0.38353 / 0.30726; rho 0.910; NATS +0.00030 ± 0.00019 / +0.00023 ± 0.00011; z -0.35 | identical | yes |
| S2 (depth 10) | 0.38225 / 0.30630; rho 0.891; NATS +0.00039 ± 0.00022 / +0.00035 ± 0.00014; z -0.24 | identical | yes |
| S4 (QuerySoftMax, KD-mix) | 0.38021 / 0.30501; rho 0.859 / 0.860; NATS +0.00074 ± 0.00030 / +0.00048 ± 0.00016; z +0.06 | identical | yes |
| S4b (QuerySoftMax, soft label) | 0.38456 / 0.30861; rho 0.882; NATS +0.00071 ± 0.00030 / +0.00043 ± 0.00015; z -0.40 | identical | yes |
| S4y (YetiRank on the teacher logit) | 0.38193 / 0.30461; rho 0.837 / 0.840; NATS +0.00058 ± 0.00027 / +0.00038 ± 0.00014; z -0.53 | identical | yes |
| bundle(S1f seed 0, hard-label tree) | 0.38457 / 0.30767; NATS +0.00098 ± 0.00035 / +0.00048 ± 0.00016; z -0.69 | identical | yes |
| bundle(3-seed bundle, hard-label tree) | 0.38493 / 0.30812; NATS +0.00093 ± 0.00034 / +0.00050 ± 0.00016; z +0.08 | identical | yes |
| bundle minus hard-label CatBoost, level | +0.0031 ± 0.0010 / +0.0034 ± 0.0006 | +0.00311 ± 0.00097 / +0.00335 ± 0.00056 (z +6.5) | yes |
| bundle minus LightGBM student, 600 rounds | -0.00022 ± 0.00071 / +0.00036 ± 0.00040 | identical | yes |
| bundle minus LightGBM student at its plateau | -0.00152 ± 0.00077 / -0.00166 ± 0.00044, z -4.08 | identical | yes |
| gap to W16, bundle | 0.0168 / 0.0166 | 0.01683 / 0.01663 | yes |
| stage-2 gate, shortfall | 0.0054 | 0.00539 | yes |
| NATS(student) minus NATS(hard-label tree), paired | -0.0004 to -0.0009 val, pooled z -1.8 to -4.2 | S4 -0.00043 / -0.00003 (z -1.76) ... S3 -0.00087 / -0.00028 (z -4.19); bundle -0.00062 ± 0.00022 / -0.00011 ± 0.00009 (z -2.77) | yes |
| S3 minus S1f at equal rounds (721) | -0.0002 ± 0.0009 / +0.0004 ± 0.0005 | -0.00018 ± 0.00087 / +0.00036 ± 0.00049 | yes |
| S2 minus S1f at equal wall time | -0.00152 ± 0.00075 / -0.00065 ± 0.00044, z -2.47 | identical (S1f reaches 721 trees at 44.1 min, S2 ran 45.0 min: curve files) | yes |
| S4b minus S1f at equal wall time | +0.0005 ± 0.0008 / +0.0012 ± 0.0005 | +0.00052 ± 0.00084 / +0.00124 ± 0.00047 (z +2.32; both 55.0 min) | yes |
| S4 and S4y end at the hard-label level | 0.3802 / 0.3050 and 0.3819 / 0.3046 | S4 minus hard -0.00129 ± 0.00102 / +0.00040 ± 0.00054; S4y minus hard +0.00043 ± 0.00088 / -0.00001 ± 0.00050 | yes |
| student bundle given [W16, hard-label tree] | +0.00007 ± 0.00010 / +0.00000 ± 0.00001 | +0.00006 ± 0.00009 / +0.00001 ± 0.00002 | yes |
| hard-label tree given [W16, student bundle] | +0.00070 ± 0.00030 / +0.00012 ± 0.00008 | +0.00068 ± 0.00029 / +0.00012 ± 0.00008 | yes |
| val M >= 50 (8,530 feeds), bundle | level 0.30557; NATS +0.00034 ± 0.00025; member -0.00009 ± 0.00041 | 0.30557; +0.00034 ± 0.00026; -0.00008 ± 0.00041 | yes |
| val M >= 50, hard-label reference NATS | +0.00090 ± 0.00041 | +0.00090 ± 0.00041 | yes |
| LightGBM students with the same tool | K: 0.38484 / 0.30761, NATS +0.00056 / +0.00038, z -1.40; KL0: 0.38613 / 0.30963, NATS +0.00054 / +0.00045, z -1.02 | identical | yes |
| trees used / built, fit minutes, selection NDCG | arm table of the report | equal to the `.curve.json` and `.eval.json` files for all ten models | yes |
| S1f plateau, 1,761 minus 1,400 trees | -0.00013 ± 0.00033 / +0.00010 ± 0.00017 | identical | yes |
| "four ways agree on val within 0.0001" | 0.38461, 0.38451, 0.38453, 0.38456; holdout "0.3080 to 0.3086" | val as stated; holdout 0.30762 to 0.30861 = 0.3076 to 0.3086 | val yes, holdout range misprinted |

Small last-digit differences (bundle member rule, joint read) come from the study re-standardising the saved z-mean file before the fit and from float32 against float64 mixing; none changes a reading.

## 2. Leak audit [M]

| question | finding |
|---|---|
| `teacher_mean11` a target only? | Yes. The nine saved `.cbm` files hold exactly the 177 feature names of record (S3: plus 128 `pe*` / `px*` columns); no teacher, out-of-fold or NN-output column. `run_s.py` passes `tr.teacher("mean11", "centred")` (or its within-feed softmax) as the label only; the logs print "label mean 0.00000 sd 1.2069". |
| tree count chosen on the train-side split, no val label? | Yes. I recomputed the click NDCG of the 13,527 `xm` selection feeds with my own code for four models at their selected tree count: 0.31165, 0.31194, 0.31110, 0.31133, equal to the curve values to five digits. The saved val / holdout predictions of six models (the three bundle members, the plateau run, S4b, S1) equal their `.cbm` applied at argmax-of-that-curve + 1 trees, max absolute difference 0. |
| fit users against selection users | 113,129 fit users, 12,697 selection users, 1 shared id: user 0, the anonymous id (6,502 feeds of `xm`), which `cbx` splits by feed index by design. |
| PCA of S3 fitted on train rows only? | Yes. Refit by my code from the `xm` train-side FIT rows (same sample rule): mean equal to 1.7e-6, all 32 directions with absolute cosine 1.000000. Label-free. It explains 0.535 of the variance on fit rows and 0.534 on the never-fitted selection feeds. |
| a holdout feed in any fit? | No. Fit set = 120,591 feeds of `xm`; `xm` and the holdout are disjoint by my own rebuild. 695 of the 28,383 holdout users (2.4%) also have feeds in the fit set: the holdout is a feed-level hold-out for every model of the project, not a famS property. |
| label-aware statistics built by famS | None. The 177 inherited columns are unchanged and identical for the reference arm (12 `tte_*` target statistics, fresh on the holdout and stale on val; 5 `pre2_*` pretrained-encoder scores) [D]; I did not re-audit them. |
| seed 1 = the 2 x T0 run cut at 886 trees: the same as an 886-tree run? | Yes for this configuration: a 20-iteration fit and a 40-iteration fit of the same seed cut at 20 trees give identical predictions (max difference 0; 3,000 feeds). |

## 3. Selection audit

- **Best arm**: chosen by the train-side selection NDCG, not on a witness. Curve files give S3 0.31200, S1f 0.31165, S1 0.31133, S4b 0.31110, S2 0.30989, S4 0.30731 [M]; rule as written (ties within 0.0005, the cheaper arm) gives S1f. The arm with the highest val / holdout level (S4b) was not taken, so the declared bundle is not a best-of-N on a witness.
- **The rule predates the fits** [D]: the run transcript shows `PREDECLARED_famS.txt` written at 01:16:36Z with the best-arm rule and the tie band, 27 seconds before the first launch (01:17:03Z); that text is byte-identical to the head of the current file. Addenda were appended at 03:13:36Z (verbatim; before S4, S4b, S2 were launched) and at 03:53:25Z (before they were read; only its time stamp was edited 8 seconds later); the decision log at 04:31:37Z. The header's own "01:25Z" is a wrong clock note, in the study's disfavour only.
- **SCREEN** applied as written, on both witnesses: the largest holdout NATS is 0.00048, less than a third of the bar.
- **MEMBER RULE** at weight 0.1 against the rebuilt W16 train-only pool: pooled z between -1.05 and +0.08 for every model and bundle. Weights 0.05 and 0.2 are reported, never chosen (bundle: +0.00004 / +0.00005 and -0.00049 / -0.00037).
- **Pilots** (15,000 feeds) were read on val and holdout and decided which variants went to full scale (disclosed). With every full-scale arm at a third to a half of the bar, this cannot have produced the negative verdict; it leaves three variants unmeasured at full scale (512-column raw block, Depthwise, Lossguide).

## 4. Process and rules

- **Read-only files** [M]: no source or data file changed after 01:10Z under `nn_x/w12/cb/{infra,audit,docs_feat,docs_rank}`, `src/`, `data/`, `subs/`, `nn_x/cb26`, `nn_x/w25`; `nats.py` is from 01:09:28Z. Files that changed under `nn/`, `nn_x/abl`, `nn_x/w5`, `preds/` are the lead's own lanes (`abl_*`, `w5_*` logs, checkpoints, predictions); the 32 W16 member files, the holdout labels and `teacher_mean11.npy` are all older than 2026-11-14 21:03Z.
- **GPU** [M]: 30 compute apps at 06:49Z (looked, nothing touched), none with famS, catboost or `run_s` in its command line; all nine models report `task_type` CPU; no famS process is alive.
- **Transcripts of the three runs** (342 shell calls) [D]: no kill, pkill, renice, install, network or submission command; no GPU setting.
- **Threads** [M]: 12 per fit in every log; never more than two 12-thread fits at once (log start and end times). A 4-thread pilot ran as a third fitting process for 82 minutes (01:25 to 02:01, 02:23 to 02:54, 02:56 to 03:12Z): the study's disclosure (a) is accurate.
- **Dry run first**: the S1 dry run (01:16:40 to 01:16:58Z) preceded its launch.
- **Third run**: started no fit; one log per tag; every read is reproducible from the files.

## 5. Attempts to break the verdict [M]

All of these are undeclared looks made to bound the negative result, never to choose anything (`logs/rv_bounds.log`).

| attempt | result |
|---|---|
| member-equal z-mean of all 10 famS models | 0.38509 / 0.30864; NATS +0.00061 / +0.00043; member z -0.13 |
| S4b + S1f plateau run + S1 (the three highest) | 0.38572 / 0.30882; NATS +0.00063 / +0.00043; member z -0.51 |
| the same three + LightGBM plateau student | 0.38666 / 0.30945 (gate short by 0.0033); NATS +0.00063 / +0.00045; member z -0.17 |
| free weights: [W16, 10 famS models] in one conditional logit (in-sample; chance level about 0.00035 / 0.00017) | +0.00121 ± 0.00039 / +0.00062 ± 0.00018 |
| free weights with the hard-label tree and both LightGBM students added | +0.00184 ± 0.00047 / +0.00073 ± 0.00020: holdout still under half the bar |
| coefficients fitted on one witness, read on the other (no in-sample optimism) | bundle +0.00035 ± 0.00017 (holdout) / +0.00050 ± 0.00023 (val); hard-label tree +0.00033 ± 0.00025 / +0.00100 ± 0.00026 |
| joint read by M bin on val | student given [W16, hard]: M >= 50 +0.00013 ± 0.00016, M < 50 +0.00001 ± 0.00006; hard given [W16, student]: +0.00068 in both bins |
| joint read with S4b in place of the bundle | S4b given [W16, hard] +0.00002 ± 0.00005 / +0.00001 ± 0.00002 |

## 6. Issues by severity

No issue changes a number or the verdict.

**Medium (wording the lead should not carry forward)**
1. "Best CatBoost student level 0.3846, reached four ways" reads like a ceiling. It is a level at the CPU caps. S1 (lr 0.1) was still rising when its 135-minute cap stopped it: 2,018 minus 1,500 trees +0.00070 ± 0.00030 / +0.00038 ± 0.00017 (z +3.2). S4b was still rising at 497 trees: 497 minus 400 +0.00095 ± 0.00034 / +0.00039 ± 0.00018 (z +3.5). Only S1f (lr 0.3) has a plateau read. The ceiling of a CatBoost student is unmeasured; the LightGBM plateau (0.3861) and the bundles of section 5 put it near 0.386 to 0.387 [I], still 0.003 or more under the gate. The study lists this under "not run" (items 3, 4, 9) but the verdict line does not say it.

**Low**
2. Verdict item 2 prints the holdout range of the four as 0.3080 to 0.3086; S1 is 0.30762, so the range is 0.3076 to 0.3086.
3. Rule 3 / 4, not in the study's own account: four numba kernel-cache files were created inside `nn_x/w12/cb/infra/__pycache__/` at 01:16:41Z and 02:31:24Z, by the first famS dry run and by the QuerySoftMax pilot, both started without `CBX_CACHE_EXTRA` (`launch.sh` and `pilot*.sh` do not export it). They are generated automatically on importing `cbx`; no existing file was changed. The study's statement "nothing written outside famS/" and its check "no file under infra changed" missed them. Harmless.
4. Rule 1: three fitting processes (28 threads) for 82 minutes in the first run. Disclosed; no effect on any number. Whether it counts as a breach is the lead's call.
5. `PREDECLARED_famS.txt` says "written 01:25Z"; the transcript says 01:16:36Z. The file's inode was re-created at 03:53:33Z by a `sed -i` that corrected a time stamp, so the file system alone cannot show the order; the transcript does (section 3).
6. Family wall time 5 h 10 min against 5 h; two agents alive on one directory from 06:01Z. Both disclosed, both without consequence.
7. Not famS-specific, for the whole track: the shared screen is safe only because `nats.py` standardises X per feed. A score that keeps its per-feed scale passes the bar without any ranking information: [W16, raw centred student score] reads +0.0037 / +0.0026, and W16 multiplied by the student's per-feed spread reads +0.0135 / +0.0111, although NDCG cannot change under a per-feed scale. In its bundle form `nats.py` does not re-standardise the z-mean; the effect is +0.00001 to +0.00005 nats here (at most 3% of the bar), so it matters only for a bundle that lands within that distance of 0.0016.

## 7. Answers for the lead

**Is the best student a candidate for stage 2?**
No. Neither condition of the plan holds: the best declared level is val 0.38461 (gate 0.3900; no re-combination of the family's files with both LightGBM students exceeds 0.38666), and no arm is PROMISING (largest NATS 0.00074 / 0.00048).
There is also nothing to feed in: no out-of-fold train-row score exists, and a student of `teacher_mean11` (in-sample on train rows) would first need five fold fits on the cross-fitted teacher.

**Does a student carry the same or different pool-independent information as the hard-label tree?**
The same, and less of it: next to the hard-label tree the student bundle adds +0.00006 ± 0.00009 / +0.00001 ± 0.00002 nats (chance level for one extra coefficient: 0.00003 / 0.00002), while the hard-label tree adds +0.00068 ± 0.00029 / +0.00012 ± 0.00008 next to the student; the same on val M >= 50, on M < 50 and for S4b.
If a tree score is ever taken to stage 2 it should be a hard-label one, not a student; that one is below the bar too (0.00117 / 0.00051), which is family B's question.

## 8. Method, files, what I did not check

Code (all in `nn_x/w12/cb/famS_review/`):
- `rv_own.py`: witnesses from raw files (val from `data/flat/val`; holdout rebuilt from `data/flat/train` and `train_prandom.npz` with the documented draw), W16 from the 16 member files, own z, NDCG and conditional logit. Imports nothing from `nats.py`, `cbx.py`, `pc_common.py`, `src/` or famS.
- `rv_reads.py` -> `logs/rv_reads.log`, `out/rv_reads.json`: every screen line, paired level reads, NATS differences, joint reads, M bins.
- `rv_bounds.py` -> `logs/rv_bounds.log`: section 5 and issue 7. `logs/rv_slope.log`: rising arms and wall-time checks. `logs/rv_bundle_scale.log`: bundle form of the screen.
- `rv_model.py` -> `logs/rv_model.log`: models against saved predictions, PCA refit, truncation diagnostic (uses `cbx` only to build pools; `CBX_CACHE_EXTRA` set to my directory).
- `rv_sel.py` -> `logs/rv_sel.log`: selection-split NDCG with own code, user overlaps.
- `logs/nats_crosscheck.log` (the shared tool, after my code agreed), `logs/gpu_apps.csv`, `logs/rv_test.log` (test files: right length, finite; bundle test file = z-mean of the three seeds to 8e-7).

Commands: `cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python -u nn_x/w12/cb/famS_review/<script>.py` (each sets 4 threads itself).

Not checked:
- The provenance of the 177 inherited feature columns (`tte_*`, `pre2_*`) and of `teacher_mean11.npy` itself; both are shared with the reference arm and with earlier studies.
- S3's saved predictions against its model (it needs the study's block builder; the arm is a null and its PCA basis was reproduced).
- The pilot numbers (15,000 feeds) and the S3 column-distribution table: not part of the verdict.
- The freshness of inputs under the test lag: famS builds no count or label statistic, so the lead's notice adds nothing beyond the inherited columns.
