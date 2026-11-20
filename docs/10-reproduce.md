# Reproducing the solution

In brief:

- The steps rebuild the first pick, FD16P20@m0.4_0.2: its recipes and seeds are in the [member table](#the-steps). The second pick differs in two places: its 16 full-view students need the lab trainer, and its popularity-light pool enters at 0.2 on every feed.
- The final pipeline is the package `avito_ranker/`, run as nine numbered scripts in 12 steps. The table gives the command, hardware and time of each step; the expected output is shown for the last three.
- The quickest check rebuilds, gates and hashes a final file from the stored member predictions in about 20 seconds per file on CPU (steps 10 to 12). The stored predictions are not in this repository, so for a reader that check follows a full run.
- Each stage was compared with the stored outputs of the original runs. Full-length training was not repeated.
- A fresh checkout has no stored output, and three commands need a change because of it ([From a fresh checkout](#from-a-fresh-checkout)).

Every command below was run on 2026-11-16 with the code as it is in this repository, except where a step says it was not re-run. Steps 10 to 12 ran again with the recipes of the final picks, on 2026-11-18 for the first and on 2026-11-19 for the second. Run times come from those re-runs or, for the long GPU stages, from the original runs; the table says which. Terms such as full-view, popularity-light, teacher and student are in [glossary.md](glossary.md).

## Environment

The runs used Python 3.12 and torch 2.11.0 built for CUDA 13.0. A training run needs one GPU with at least 55 GiB free. The original runs had a few NVIDIA data-centre GPUs; the model and the number are not given here. The original runs shared each GPU among several processes; the checks of 2026-11-16 ran one process per GPU, with 12 CPU threads per process. The lock file pins every version:

```
uv sync                      # package, tests, figure code and the lab's libraries
export OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
```

The CPU scripts (01, 02, 03, 08, 09 and `freeze_check.py`) set the polars and numba caps themselves from `--threads`, default 12. The export covers the GPU scripts and the tests.

`uv sync` installs `avito_ranker` in editable mode, which the scripts and tests need. With another installer, run from the repository root with `PYTHONPATH=.`.

A non-editable install does not carry `configs/paths.toml`. In that case point `AVITO_PATHS` at a copy of it and set the copy's `root` entry to the checkout. Relative entries resolve against `root`, `root` resolves against the folder of the paths file, and the recipes are found through the `[configs]` entries of the same file.

## Data and outputs

Every location of data and outputs comes from `configs/paths.toml`. It points into `artifacts/`, which is not versioned; the competition data is not part of this repository. The competition archive (17.9 GiB) and the sample submission go to `artifacts/data/raw/`, the extracted parquet and embedding tables to `artifacts/data/competition/`.

No script replaces a stored artefact with other values. It refuses, and `--out-dir` (or `--out-file`) writes a rebuild elsewhere for comparison. The builders of steps 1 to 3 leave an array in place when the rebuild gives the same values, so a run against the stored arrays checks them. `--overwrite` replaces one that differs, after unlinking it, so that a hard-linked copy keeps its stored values.

## The steps

The steps run in the order of the table, which is not the numeric order of the scripts. `02_build_features.py` runs before and after `03_train_encoder.py`, and `04_train_member.py` runs for the teacher members before `05_build_teacher.py` and for the students after it. The last column says whether a time was measured on a re-run or comes from the original runs.

| Step | Command | Hardware | Time | Measured |
|---|---|---|---|---|
| 1. Flatten the parquet | `python scripts/01_flatten.py` | CPU, 12 threads | 17 s for train, under 1 s for validation and test | re-run on 2026-11-16 |
| 2. Inputs without the encoder | `python scripts/02_build_features.py counts covisitation context exposure exploration` | CPU, 12 threads, 43 GB | 33 min on all three splits: co-visitation 17 min, exploration 11 min, feed context 4 min, counts and exposure under a minute each | re-run on 2026-11-16 |
| 3. Trajectory encoder | `python scripts/03_train_encoder.py corpus`, then `train <model>` and `score <model>` for the ten models | CPU, then one GPU per model | corpus 2 min (writes 79 GB); training 13 to 59 min per model, 5.2 GPU-hours for all ten; scoring 1 to 3 min per model, 63 GB of GPU memory | original run for training; re-run on 2026-11-16 for corpus and scoring |
| 4. Encoder inputs and fine-tune extras | `python scripts/02_build_features.py encoder extras` | CPU, 12 threads | 31 s for the encoder inputs, 2 min for the extras | re-run on 2026-11-16 |
| 5. Click-label teacher members | `python scripts/04_train_member.py configs/recipes/teacher_hard_label.toml --seed S --member NAME`, for 12 of the 25 members; the other 13 cannot be retrained with the package ([Gaps in the lineage](#gaps-in-the-lineage)), and step 6 reads their stored train scores as they are | one GPU, 55 GiB free | 57 to 182 min each on GPUs shared by several jobs | original run |
| 6. Teacher files | `python scripts/05_build_teacher.py dump CKPT ... --out-file artifacts/teachers/teacher_<group>.npy`, then `average configs/recipes/teacher_mean11.toml` and `teacher_mean25.toml` | one GPU, then CPU | 105 s per member plus about 1 min per input view to load the train split; 10 s per average | re-run on 2026-11-16 |
| 7. Students | `python scripts/04_train_member.py configs/recipes/student_wide.toml --seed S --member NAME`, for the students of pick 1; the 16 full-view students of pick 2 need the lab trainer (member table below) | one GPU, 55 GiB free | 221 to 259 min each | original run |
| 8. Train-only predictions | `python scripts/06_predict.py --checkpoint FILE --member NAME` (the trainer already writes them for new runs) | one GPU | 11 s per member | re-run on 2026-11-16 |
| 9. Fine-tune on the validation days | `python scripts/07_finetune.py --checkpoint FILE --member NAME` | one GPU, 5 GB | 4 min per member (3 seeds × 5 folds) | re-run on 2026-11-16 |
| 10. Ensemble and submission file | `python scripts/08_build_submission.py configs/ensembles/fd16p20_m04_02.toml --write`, then the same with `fdx16p20_ps.toml` | CPU, 12 threads, 4 GB | 11 to 12 s per file | re-run on 2026-11-18 and 2026-11-19 |
| 11. Pre-submission gate | `python scripts/09_check_submission.py configs/ensembles/fd16p20_m04_02.toml`, then the same with `fdx16p20_ps.toml` | CPU, 12 threads | 4 s per file | re-run on 2026-11-18 and 2026-11-19 |
| 12. Freeze-day check | `python scripts/freeze_check.py configs/ensembles/fd16p20_m04_02.toml configs/ensembles/fdx16p20_ps.toml` | CPU, 12 threads | 4 s per file | re-run on 2026-11-19 |

The train split stays on the GPU during training: 36.8 GiB for the full view and 33.6 GiB for the popularity-light view once loaded, with peaks of 52.5 and 46.0 GiB while it loads (`avito_ranker/data/loader.py`).

The steps map to the nine stages of the [pipeline figure](figures/pipeline.svg) as follows. Step 12, the freeze-day check, has no stage of its own.

| Stage of the figure | Steps |
|---|---|
| 1. Flatten | 1 |
| 2. Exploration probability | 2 |
| 3. Candidate inputs with the trajectory encoder | 2, 3 and 4 |
| 4. Click-label rankers for the teacher | 5 |
| 5. Teacher files | 6 |
| 6. Students | 7 and 8 |
| 7. Fine-tune on the validation days | 9 |
| 8. Ensemble and file | 10 |
| 9. Pre-submission gate | 11 |

The members of each final file and their recipes:

| File | Members | Recipes and seeds |
|---|---|---|
| FD16P20@m0.4_0.2 (first pick) and FD16P20 | the same 16 full-view students and 4 popularity-light students, all width 192, all trained on every train feed; `fd16p20_m04_02.toml` weights the popularity-light pool 0.4 on feeds of fewer than 50 candidates, `fd16p20.toml` 0.2 on every feed | `student_wide` and `student_wide_t25`, seeds 951 to 958; `student_popularity_light`, seeds 951 to 954 |
| FD16P20vf | FD16P20 plus 4 popularity-free students | add `student_popularity_free`, seeds 951 to 954 |
| FDX16P20:ps (second pick) | 16 pseudo-feed full-view students (width 192), trained on every train feed plus pseudo-feeds built from user histories, and the 4 popularity-light students of FD16P20; `fdx16p20_ps.toml` weights the popularity-light pool 0.2 on every feed | full-view students: no package recipe; the lab trainer `lab/nn_x/w12/pseudokd/train_full_pk.py`, launched by `lab/nn_x/w12/pseudokd/full/lane_fps.sh`, seeds 951 to 958 for each teacher; `student_popularity_light`, seeds 951 to 954 |
| W8P20 (second pick until 2026-11-19) | 8 full-view students (width 192) and 4 narrow popularity-light students (width 128), each holding out 30,000 train feeds | `student_wide_holdout`, `student_wide_t25_holdout`, `student_popularity_light_narrow_t25_holdout`, seeds 951 to 954 |

The package trainer has no pseudo-feed option. The lab launcher runs the trainer with the arguments of the full-view students of FD16P20 plus `--pseudo_ratio 1.0 --pseudo_w 0.3 --pseudo_head 1 --pseudo_gap_min 16 --pseudo_gap_max 64 --pseudo_tmax 63`, and every argument is in the members' published read-outs, `lab/nn_x/w5/logs/<member>.json` (key `args`). Like other lab scripts it needs the old layout (`lab/tools/legacy_layout.sh on`) and the stored arrays it reads.

## Where new runs write and what steps 10 to 12 read

Steps 5 to 9 write into `artifacts/runs/`: `--out-dir` of 04, 06 and 07 names that run folder, with `checkpoints/`, `logs/` and `predictions/` inside.

Steps 10 to 12 read the stored members by default. The builder reads member predictions from `outputs.predictions` and the gate reads logs from `outputs.member_logs`; both print the folders they read first.

To build and gate a file from new runs, pass `--members-dir artifacts/runs/predictions` to step 10 and `--logs-dir artifacts/runs/logs` to step 11. The gate fails a member whose input list is not in that folder. The reference pools the gate compares against (`ens_W8P20kd_*` for the all-data files, `ens_W8kd_*` for W8P20) are read from `outputs.predictions` in every case.

Steps 10 to 12 need only the stored member predictions (80 arrays for FD16P20@m0.4_0.2, the same 80 for FD16P20, 80 for FDX16P20:ps of which 16 are shared with FD16P20, 96 for FD16P20vf, 48 for W8P20), the flat validation and test arrays, the sample submission and the member logs. That is the quickest way to check a final file.

## From a fresh checkout

The commands of the table were run on a machine that holds the outputs of the original runs. A reader starts without them, and three steps then stop as written. This was established by reading the code; no run from an empty tree was made.

- **Step 2** reads a stored model of the display position and stops when it is missing. Add `--refit-proxy` to fit a new one. A refit can move the exploration probability of some feeds, and with it the holdout and the cooldown pool ([Refitting the exploration probability](#refitting-the-exploration-probability)).
- **Step 6** averages group files of all 25 teacher members and stops on a missing group. Only 12 members can be trained with the package: `w5_e10_base_s951` to `s954` and `abl_e10_pop1_s951` to `s958`. Copy the two teacher recipes, keep their `name`, and list only the groups dumped from those members. The 11-member teacher then has 4 members and the 25-member teacher 12. How far that moves the students was not measured.
- **Step 11** compares the new file with the stored pools of W8P20, read from `outputs.predictions` even with `--members-dir`. Build those pools first with step 10 on `w8p20.toml`, which needs its 12 held-out students. Or copy the ensemble config and set `reference_pool_tag` under `[gate]` to an earlier ensemble built with `--write`.

Steps 10 and 11 also take `--members-dir artifacts/runs/predictions` and `--logs-dir artifacts/runs/logs`, as the section above says. A run made this way gives a file of the same recipe with a weaker teacher. It does not give the submitted file, and its checksum will differ from the ones below.

## What to expect

Step 10 prints the folders it read, the validation read-out and the checksum of the file it writes:

```
member predictions from artifacts/predictions
reference pools from artifacts/predictions
FD16P20@m0.4_0.2: 16 full-view + 4 popularity-light members, popularity-light weight 0.4 on feeds of fewer than 50 candidates, 0.2 on the others
val NDCG: final 0.40293 | fine-tuned branch 0.40139 | train-only branch 0.40208
sha256 3cf6faccbad1357ae1a1b11999fef8168835d77407684832c7578c832c6e71b3  cand_FD16P20_m04_02kd_r4.parquet
```

The submitted file is not in this repository; the checksum is the one recorded for it, so a rebuild that prints the same value has reproduced the file.

For the second pick (`fdx16p20_ps.toml`) the read-out is `val NDCG: final 0.40286 | fine-tuned branch 0.40194 | train-only branch 0.40234` and the checksum starts with `8235c55c`. W8P20 (`w8p20.toml`), the second pick until 2026-11-19, reads 0.40214 and gives `406e5705`. The two earlier candidates have their own recipes: `fd16p20.toml` reads 0.40271 and gives `14adc19a`, and `fd16p20vf.toml` reads 0.40286 and gives `4a826d6c`.

Step 11 prints the folders it read on standard error and the gate text on standard output. For the five stored files the text equals `docs/records/gate-outputs/gate_<pool tag>.txt` ([records/gate-outputs/](records/gate-outputs/)) byte for byte and ends with `GATE: PASS  (FAIL 0, WARN 0)`. It exits with code 1 on any failure. Step 12 ends with `freeze check passed at <UTC time>`.

## How the package was checked against the original runs

The package was written after the competition work, from the lab code that produced the submissions. Each stage was compared with the stored outputs of the original runs.

| Stage | Compared with | Result |
|---|---|---|
| Ensemble and submission writer | the three candidate files of 2026-11-16, rebuilt from 224 stored member predictions; on 2026-11-18 also the first pick, from the 80 predictions it shares with FD16P20; on 2026-11-19 the second pick, from its 80 stored member predictions | sha256 identical for all five; the 20 pool files byte-identical |
| Pre-submission gate, private-mix view, calibration | the stored gate outputs, the old tools' weights | gate text byte-identical for all five files; weights bit-equal |
| Model, loader, prediction | all 36 member checkpoints of the three files of 2026-11-16 (the 16 pseudo-feed students of the second pick were not re-scored), scoring validation (1,389,971 rows) and test (2,904,806 rows) | bit-identical to the stored predictions (largest difference 0.0); parameter counts 2,118,848, 2,114,613 and 1,005,685 as recorded |
| Flattening | the stored arrays of all three splits | identical |
| The 34 inputs, the exposure input, the 42 fine-tune extras | the stored arrays on train (246,745,836 rows), validation and test | 155 of 155 arrays bit-identical |
| Exploration probability | the stored file for 3,151,208 train feeds | bit-identical when built from the stored position model (see below) |
| Trajectory encoder | stored corpora and the 2.1 billion stored window scores | corpora and scores bit-identical; a 432-step training run matches two runs of the original trainer within their own spread (see below) |
| Trainer | the original trainer, seeded runs of 246 to 300 steps for the full-view, popularity-light and click-label recipes, with and without the cooldown | with deterministic kernels every step is bit-identical (loss, gradient norm, batch order, learning rate, final weights); with the production kernels the two differ by 5e-5 relative per step, the same as the original trainer against itself |
| Teacher dump and average | four stored dump files on all train rows; the two stored teacher files | dumps bit-identical; averages within 3.3e-5 (11 members) and 4.8e-7 (25 members) |
| Fine-tune | the original fine-tune, one seed, one full-view and one popularity-light member; the full 3-seed recipe against the stored predictions | validation NDCG within 0.0001, the same as two runs of the original; full recipe within 0.00007 of the stored predictions, prediction correlation above 0.99999 (see below) |

Not re-run after the move to the package: a full-length training of any member, the ten encoder trainings, the three-seed fine-tune of all 36 members, and the scoring of the 16 pseudo-feed students of the second pick from their checkpoints.

A full re-run on GPU would not return the stored checkpoints anyway. The backward pass on GPU is not bit-reproducible under the production settings: atomic additions in the backward kernels sum gradients in an order that changes from run to run. A re-trained member therefore lands within the seed spread, not on the stored weights. With `torch.use_deterministic_algorithms(True)`, the math attention kernel and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, a seeded run is bit-reproducible, at some cost in speed.

### Fine-tune: package against the lab code

The package and the lab's fine-tune were compared on one full-view and one popularity-light member, with one seed per fold, on the GPU model that trained them. Their out-of-fold validation NDCG differed by 1.0e-4 and 3e-6. The within-feed z-scores of the two runs correlated at 0.999997 on validation and 0.9999994 on test.

Two runs of the lab code differ from each other by as much: 7.7e-5 and 2.7e-5 in NDCG, with the same correlations. With the production recipe the package gives 0.39936 and 0.39921 out of fold, against 0.39934 and 0.39928 for the stored predictions the ensembles were built from.

### Trajectory encoder: package against the lab code

The stored checkpoints load into the package's encoder and reproduce the stored scores bit for bit with the recipe's chunk sizes. The training loop ran for 432 steps of the `all_id64` settings beside the original script. After 400 steps the feed-open loss was 8.298 and 8.339 in two runs of the original and 8.325 with the package. Training on the GPU is not run-to-run reproducible in either version, so this is as close as the comparison gets.

## Gaps in the lineage

Three pieces of the original lineage did not survive as code:

- The script that averaged the teacher logits was never saved. The package's averaging is new and was checked against the stored teacher files as above.
- The trainer source that produced the 8 full-view students of W8P20 was edited the next day. The package's recipes match those students' logged arguments, and W8P20 rebuilds from its stored predictions, but not from identical source.
- The 4 narrow popularity-light students of W8P20 had no launcher on disk. Their command survives only in their logs, which is where the recipe comes from.

The second pick adds a gap of another kind. Its 16 full-view students come from a lab trainer with pseudo-feed flags that the package trainer does not have. That code is in the repository (`lab/nn_x/w12/pseudokd/`), so these members can be retrained with it under the old layout, but not with the package. Their checkpoints were kept in the lab's own checkpoint folder, outside `artifacts/members/`, and like every checkpoint they are not published.

Thirteen of the 25 teacher members cannot be retrained with the package. Seven were short cooldown re-runs of an older ranker, with no recipe in `configs/` and no stored checkpoint, and six use a recurrent (GRU) block over the history that the package does not include. Their summed train scores are stored, and step 6 averages them as they are.

### Refitting the exploration probability

The exploration probability, the estimated chance that a train feed was shown in random order, comes from a small LightGBM model of the production display position. Step 2 reads the stored model by default, which reproduces every stored value. `--refit-proxy` refits it, and a refit reproduces the stored file only by luck.

At each split LightGBM decides where missing values go by comparing the gain of sending them left with the gain of sending them right. When the node held no missing value in training, the two gains are the same number reached along two summation paths, and the last bits decide. Those bits depend on the order in which gradients are added up, which depends on the thread count, the histogram layout and, from run to run, on thread timing.

The training rows are unaffected, but at scoring time candidates with a missing input reach such splits. `hsim_max`, for example, is missing for 1.2% of the candidates the proxy is fitted on.

`PROXY_PARAMS` in `avito_ranker/features/exploration.py` keeps the settings of the stored fit: 40 threads, as its model file records, and row-wise histograms, which LightGBM had picked by a timing test. Two forced row-wise refits repeated the first 16 and 18 stored trees bit for bit; a column-wise one differed from the first tree.

The per-feed statistic `z` is the rank correlation between the proxy's prediction and the shown position, scaled to be standard normal under random order. On the identical training matrix, every refit that changed `z` had flipped the same split, on `hsim_max` in the first tree. Two of three refits with these settings kept it and reproduced `z` and the probability of every train feed bit for bit; the third flipped it, as both 12-thread refits did.

A refit that flips the split changes:

- `z` on 2.64M of the 2.75M scored feeds and the probability on 1.84M; 4.3% of the scored feeds move by more than 0.1.
- The auxiliary-loss weight of the trainer by more than 0.1 for 3.1% of the feeds with 50 or more candidates.
- The 30,000-feed holdout, which keeps only 9,340 of its feeds, since its random draw runs over a slightly different candidate list.
- The pool of random-order feeds replayed in the cooldown (probability above 0.5), which keeps 155,973 of its 164,118 feeds.

## Tests

```
python -m pytest -m "not artefacts and not gpu"   # 179 tests, no data needed, about 7 min
python -m pytest -m "artefacts and not gpu"       # 28 tests on the stored artefacts, about 4 min
python -m pytest -m gpu                           # 6 tests on one GPU, about 1 min
```

Without the data or a GPU, the 34 marked tests skip. Two of the artefact tests also skip on the stored tree: they compare the teacher recipes with the logs of the click-label teacher members, which are not stored.
