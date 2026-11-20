#!/usr/bin/env bash
# watch_growth2.sh: after the 16 new all-data wide students s959-966 are fine-tuned: watch_growth.sh FD32P20 959 966 (acceptance, build, gate, PAIR read), then build + gate the DISJOINT
# all-data pick-2 design FDB16P20 (16 new wide + pop1 wide s955-958) and print its PAIR reads against FD16P20 and FD32P20. CPU only. Ledger 2026-11-15T23:08Z (S3, S4).
cd <repo> || exit 1; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
bash nn_x/w12/kd/fd/watch_growth.sh FD32P20 959 966
echo "=== $(date -u +%FT%TZ) FDB16P20 ==="
bash nn_x/w12/kd/fd/fd_finish.sh FDB16P20 2>&1 | tail -n 14 | cut -c1-400
.venv/bin/python nn_x/w12/kd/build_r4.py FD16P20,FDB16P20 2>&1 | grep -E "^PAIR|final val" | cut -c1-520
.venv/bin/python nn_x/w12/kd/build_r4.py FD32P20,FDB16P20 2>&1 | grep -E "^PAIR|final val" | cut -c1-520
