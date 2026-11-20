## 9. Exact commands

Environment of every fit (`launch.sh`): `cd <repo> && export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12`, then
`setsid nohup .venv/bin/python -u nn_x/w12/cb/famS/run_s.py --tag <tag> <args> > nn_x/w12/cb/famS/logs/<tag>.log 2>&1 < /dev/null &`

| tag | arguments of `run_s.py` |
|---|---|
| `s1_rmse_d8_lr010_s0` | `--lr 0.1 --iters 3000 --seed 0 --budget_min 135` |
| `s1f_rmse_d8_lr030_s0` | `--lr 0.3 --iters 1000 --seed 0 --budget_min 55` |
| `s3f_pca_d8_lr030_s0` | `--lr 0.3 --iters 1000 --seed 0 --block pca --budget_min 75` |
| `s2_rmse_d10_lr030_s0` | `--depth 10 --lr 0.3 --iters 1000 --seed 0 --budget_min 45` |
| `s4_qsm_kdmix_lr030_s0` | `--loss qsm --kd_w 0.5 --lr 0.3 --iters 1000 --seed 0 --budget_min 35` |
| `s4b_qsm_soft_lr030_s0` | `--loss qsm --kd_w 1.0 --lr 0.3 --iters 1000 --seed 0 --budget_min 55` |
| `s1f_rmse_d8_lr030_s1long` | `--lr 0.3 --iters 1772 --seed 1 --budget_min 112` |
| `s1f_rmse_d8_lr030_s2` | `--lr 0.3 --iters 886 --seed 2 --budget_min 70` |
| `s4y_yeti_teacher_lr030_s0` | `--loss yeti --lr 0.3 --iters 400 --seed 0 --budget_min 33` |

Chained starts (a waiter on the PID of one of this study's own processes, never a signal): `nn_x/w12/cb/famS/chain_after.sh <pid> <tag> <args>`.
Dry runs (3,000 feeds, 40 trees, 4 threads) of every arm type: `CBX_THREADS=4 ... run_s.py --tag dry_<x> ... --iters 40 --max_feeds 3000 --out nn_x/w12/cb/famS/dry`.
Pilots: `pilot.sh`, `pilot2.sh`, `pilot3.sh` (15,000 feeds, 4 threads, `--out nn_x/w12/cb/famS/pilot`).

Reads (4 threads, evaluation only):
```bash
cd <repo> && export CUDA_VISIBLE_DEVICES="" CBX_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 POLARS_MAX_THREADS=4
F=nn_x/w12/cb/famS
.venv/bin/python $F/fs_read.py nats <tag> [<tag> ...]                 # the nats.py line (same function, tags resolved to files)
.venv/bin/python nn_x/w12/cb/nats.py <tag> $F/preds/<tag>_val.npy $F/preds/<tag>_hold.npy      # the shared tool itself, same output
.venv/bin/python $F/fs_read.py bundle bundle_s1f_3seeds s1f_rmse_d8_lr030_s0 s1f_rmse_d8_lr030_s1 s1f_rmse_d8_lr030_s2
.venv/bin/python $F/fs_read.py cmp <tagA> <tagB>                      # paired per-feed NDCG difference
.venv/bin/python $F/fs_read.py joint <student tag> ts_lr020_s0        # section 4
.venv/bin/python $F/fs_read.py mbin <tag>; .venv/bin/python $F/fs_m50.py <tag>          # section 5
.venv/bin/python -u $F/fs_staged.py <tag> <none|pca> 100,200,300      # predictions at fixed tree counts -> staged/<tag>@<n>_{val,hold}.npy
.venv/bin/python $F/fs_curve.py <tag> [--at 100,200,...]              # train-side selection curve
.venv/bin/python -u $F/fs_trunc.py s1f_rmse_d8_lr030_s1long s1f_rmse_d8_lr030_s1 886 none      # seed 1 at the tree cap of seed 0
.venv/bin/python $F/fs_bundle.py bundle_s1f_3seeds_z s1f_rmse_d8_lr030_s0 s1f_rmse_d8_lr030_s1 s1f_rmse_d8_lr030_s2   # saves the bundle as files
.venv/bin/python -u $F/fs_dist.py pca                                 # label-free column distributions
$F/fs_final.sh                                                        # all final reads in one go (log: logs/final_reads.log)
```

## 10. Files (all under `nn_x/w12/cb/famS/` unless stated)

- Code: `run_s.py` (arm runner on `cbx.run_arm`), `fs_common.py` (embedding blocks, KD label), `fs_read.py`, `fs_m50.py`, `fs_staged.py`, `fs_curve.py`, `fs_trunc.py`, `fs_bundle.py`, `fs_dist.py`, `fs_table.py`, `fs_lines.py`, `launch.sh`, `chain_after.sh`, `pilot*.sh`, `fs_final.sh`.
- Pre-declaration and state: `PREDECLARED_famS.txt` (01:25Z; addenda 03:15Z and 03:53Z; decision log 04:31Z), `PROGRESS.md`, `LEAD_NOTICE.md`.
- Predictions (float32; val / test in the row order of `preds/`, holdout in the row order of `nn_x/w5/preds/holdout`): `preds/<tag>_{val,hold,test}.npy`, with `<tag>.eval.json`, `<tag>.curve.json`, `<tag>.cbm` for every tag of section 9; the bundle as per-feed z-mean in `preds/bundle_s1f_3seeds_z_{val,hold,test}.npy` (+ `.members.json`).
- Reads: `reads/nats_<tag>.json`, `reads/cmp_<a>__<b>.json`, `reads/joint_<s>__ts_lr020_s0.json`, `reads/mbin_<tag>.json`, `reads/m50_<tag>.json`, `reads/dist_pca.json`, `reads/cmp_lgb_students.json`; staged predictions `staged/`; pilots `pilot/`; dry runs `dry/`; logs `logs/`.
- Cache: `cache/pca32.npz` (PCA basis of S3; nothing else was cached: the blocks are rebuilt in RAM in 15 s).
- NOT produced: out-of-fold tree scores for TRAIN rows (stage 2 would need them: 5 fold fits of the chosen recipe); any file outside `nn_x/w12/cb/famS/` except this report.
