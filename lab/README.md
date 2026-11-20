# The lab tree

This folder is a trimmed copy of the working tree in which the research ran, from 2026-11-11 to 2026-11-19. Everything in `docs/`, `experiments/` and `avito_ranker/` was distilled from it. It is an archive to look things up in, not code to run.

What is here: the code as it ran, the written reports and the per-run read-outs, as JSON and text. What was left out of the publication:

- run logs of every kind, arrays, predictions, checkpoints and old model dumps;
- launch, queue and wait scripts, apart from 17 run scripts that a page cites, and their markers, progress files, command lists and dry-run outputs;
- the working records: the status log, the decision ledger, the experiment registry, the coverage map, the freeze protocol and the picks audit;
- samples of the competition data, third-party code, and other participants' names, personal data and secrets.

Details of the original machine are redacted in what remains: a device number reads `N`, a list of them `<ids>`, a memory figure `[n] GB`, and the path of the repository on that machine `<repo>`. Remarks about other jobs on the machine were neutralised.

So a number in the docs can be traced to its script and its read-out, not to its log. [docs/evidence.md](../docs/evidence.md) does this for the headline numbers.

Expect it to read like a lab notebook. Directory names are dated or numbered by round (`nn_x/w5`, `nn_x/r6`, `reports/STALE_1115`), the same trainer exists in many slightly different copies, and run names are code names ([docs/glossary.md](../docs/glossary.md)).

## Where things are

| Path | Files | What it holds |
|---|---|---|
| `nn/` | 476 | The first neural ranker of 2026-11-11 and one JSON read-out per run in `nn/logs/` |
| `nn_x/` | 7,109 | Every experiment after the first day, one folder per round or study. `nn_x/w5/` is the trainer harness of all distilled students, with one JSON read-out per member in `w5/logs/`. `nn_x/abl/` holds the popularity-light and popularity-free students. `nn_x/w12/` holds the work of 14 to 19 November: distillation in `kd/`, all-data retrains in `stale/fulldata/`, pseudo-feed students in `pseudokd/`, the freeze tools in `freeze/` |
| `reports/` | 465 | Written reports of the studies (183 markdown files), with their scripts and small result files |
| `eda/` | 229 | The analyses of the first night: splits, embeddings, collaborative signals, display position, the later-snapshot leak, errors |
| `solvers/` | 129 | CPU studies of rounds 5 to 7: ensembling, errors, stale replay |
| `feats_v2/`, `feat_new/` | 141 | Builders of the pre-computed candidate inputs |
| `pretrain/` | 27 | The trajectory encoder and the scripts that turn its scores into inputs |
| `src/` | 23 | The shared library of the first days: flattening, leave-user-out counts, boosted trees, the submission writer, the gate |
| `logs/` | 38 | JSON read-outs of 2026-11-11, including the encoder runs |
| `archive/` | 57 | Superseded scripts and gate read-outs of the first days |
| `tools/` | 1 | `legacy_layout.sh`, described below |

The whole folder is about 8,700 files and 108 MB. It holds 2,971 Python files, 3,835 JSON read-outs and 17 shell scripts.

The final pipeline's lineage runs through `src/prep.py`, the input builders in `feats_v2/`, `feat_new/`, `pretrain/` and `src/features*.py`, the trainers `nn_x/w12/stale/fulldata/train_full.py` and `train_full_abl.py` (harness code in `nn_x/w5/` and `nn_x/abl/`), the fine-tune `nn_x/w5/finetune.py`, and the builder `nn_x/w12/kd/build_r4.py`. The 16 full-view students of the second pick come from `nn_x/w12/pseudokd/train_full_pk.py`.

## Where a number comes from

- A member's training: `nn_x/w5/logs/<member>.json` (arguments, validation curve, minutes) and, for members trained on all data, `<member>_fulldata.json` (feeds trained on, step kept). Popularity-light members are in `nn_x/abl/logs/`.
- A study: its folder under `nn_x/` or `reports/` holds the script, an `out/` or `logs/` folder with JSON or text read-outs, and often a written report.
- The 312 catalogued ideas: the "Where" column of [docs/07](../docs/07-what-failed.md#the-catalogue-by-theme) names the script or read-out of each row.

## Running a lab script

The scripts do not run from this tree as it is. 1,814 of the 2,971 Python files spell absolute paths of the original machine, shown here as `<repo>`, and most read arrays, predictions or checkpoints that are not published.

A run needs the competition data, the outputs of the earlier stages the script reads, and the old layout. The package rebuilds the stages of the final pipeline ([docs/10](../docs/10-reproduce.md)). For the layout:

```
lab/tools/legacy_layout.sh on      # top-level symlinks: nn -> lab/nn, data -> artifacts/data, preds -> artifacts/predictions, ...
cd lab/nn_x/w5 && ...              # run the script as its header describes
lab/tools/legacy_layout.sh off     # removes exactly those symlinks
```

Absolute paths have to be restored: replace `<repo>` by the path of your checkout, and the redacted device numbers by your own.

## Things to know

- Apart from the removals and redactions above, nothing was cleaned. Failed runs, abandoned scripts and superseded reports are left in place; the docs say which results stood.
- Many files still name the removed logs and records by their old paths, for example `reports/CURRENT_STATUS.md`. Those paths lead nowhere in this copy.
- Reports here were written during the work. Where a report and a page under `docs/` disagree, the page is the later reading.
- Sixteen text files contain Russian: organiser material, a decoding of category names and a few read-out lines.
- The members of the final files are listed in `configs/ensembles/`. Their predictions and checkpoints are not published.
