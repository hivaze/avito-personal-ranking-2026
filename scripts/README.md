# Scripts

In brief: the pipeline as numbered steps 01 to 09, plus `freeze_check.py`, the check run on the final files before they are selected. The table gives each script's job, hardware and run time. [docs/10-reproduce.md](../docs/10-reproduce.md) gives the order to run them in, with expected outputs.

## Running the steps

Run the scripts from the repository root with the package installed (`uv sync`) or importable (`PYTHONPATH=.`). Every location comes from `configs/paths.toml`, or from the file named by `AVITO_PATHS`. Each script's `--help` shows example commands, run time and hardware.

A stored artefact is never replaced by other values. A script refuses, and `--out-dir` writes a rebuild elsewhere.

The scripts do not run in numeric order: 02 runs before and after 03, and 04 before and after 05. Terms such as member, teacher and pre-submission gate are defined in [docs/glossary.md](../docs/glossary.md).

| Script | What it does | Runs on | Time |
|---|---|---|---|
| `01_flatten.py` | competition parquet to flat arrays per split | CPU, 12 threads | 17 s for train |
| `02_build_features.py` | candidate inputs, exposure input, exploration probability, encoder inputs, fine-tune extras | CPU, 12 threads | 1 to 17 min per family |
| `03_train_encoder.py` | trajectory-encoder corpora, training, raw scores | CPU, then one GPU | 2 min; 13 to 59 min per model; 1 to 3 min per model |
| `04_train_member.py` | trains one ensemble member from a recipe | one GPU, 55 GiB | 180 to 259 min |
| `05_build_teacher.py` | sums the train scores of click-label members, averages them into a teacher | one GPU, then CPU | 105 s per member plus about 1 min per input view; 10 s |
| `06_predict.py` | scores validation and test with a trained member | one GPU | 11 s per member |
| `07_finetune.py` | out-of-fold fine-tune on the validation days | one GPU | 4 min per member |
| `08_build_submission.py` | pools the members, prints the validation read-out, writes the submission file | CPU, 12 threads | 11 s |
| `09_check_submission.py` | the pre-submission gate | CPU, 12 threads | 4 s |
| `freeze_check.py` | checksum and gate re-run of the files about to be selected, compared with the record | CPU, 12 threads | 4 s per file |
| `export_catalogue.py` | writes the catalogue of 312 ideas in `docs/07-what-failed.md` as the table `docs/catalogue.csv`; no data needed | CPU | under 1 s |

## Checking the final files

The recipes of the two final files are `configs/ensembles/fd16p20_m04_02.toml` (pick 1, FD16P20@m0.4_0.2, public 0.3952092) and `configs/ensembles/fdx16p20_ps.toml` (pick 2, FDX16P20:ps, public 0.39526061). `configs/ensembles/w8p20.toml` keeps W8P20, pick 2 until 19 November. [docs/11-final-picks.md](../docs/11-final-picks.md) gives the reasons for the two files.

- `python scripts/freeze_check.py configs/ensembles/fd16p20_m04_02.toml configs/ensembles/fdx16p20_ps.toml` checks both against the record.
- `python scripts/09_check_submission.py configs/ensembles/fd16p20_m04_02.toml` re-runs the gate on pick 1 and prints the text stored when the file was built ([docs/records/gate-outputs/](../docs/records/gate-outputs/)).

## Options

Options mean the same thing in every script that takes them.

A required recipe, a TOML file from `configs/recipes/` or `configs/ensembles/`, is the first positional argument of 04, `05 average`, 08, 09 and `freeze_check.py`. Where the recipe has a default (the `encoder` family of 02, 03 and 07), `--recipe FILE` replaces it. `--member NAME` and `--seed N` name a training run, `--checkpoint FILE` points at a member checkpoint and `--splits` chooses the splits.

`--out-dir DIR` says where new outputs go. For 04, 06 and 07 it is the run folder, with `checkpoints/`, `logs/` and `predictions/` inside (default `outputs.runs`, that is `artifacts/runs/`). For 01 and 02 it is the folder of the arrays, for 08 the folder of the five files it writes and for `freeze_check.py` the folder of the day's gate texts. `--out-file` names the single output of 05.

By default the steps after training read the stored members. 08 reads member predictions from `outputs.predictions`, and 09 and `freeze_check.py` read member logs from `outputs.member_logs`. Each prints the folders it read as its first lines.

Steps 4, 6 and 7 write to `<outputs.runs>/predictions` and `<outputs.runs>/logs`. A rebuild from new runs therefore passes `--members-dir artifacts/runs/predictions` to 08 and `--logs-dir artifacts/runs/logs` to 09. `--in-dir` names the folder of a rebuilt submission and its pool files for 09. A member whose input list is missing from the logs folder fails the gate.

`--device` picks the GPU (03 to 07). `--threads` sets the CPU threads of polars and numba (01, 02, 03, 08, 09 and `freeze_check.py`, default 12).

The builders of steps 1, 2 and 3 leave an array alone when a rebuild gives the same values, so a run against the stored arrays checks them. `--overwrite` replaces an array that holds other values. It unlinks the old file first, so a hard-linked copy keeps its stored values.

## Figures

`scripts/figures/` holds the code that draws `docs/figures/`. `python scripts/figures/make_all.py` redraws all ten figures in about 2 s. `python scripts/figures/count_catalogue.py` recounts the catalogue of `docs/07-what-failed.md` into `docs/figures/data/experiments_outcomes.csv`. Run it after every edit of the catalogue. With `--check` it only reports.
