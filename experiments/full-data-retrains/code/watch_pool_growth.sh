#!/usr/bin/env bash
# Read of a larger pool. Waits until the new all-data students of a seed range are fine-tuned, runs the
# acceptance, builds and gates the larger file, and prints its paired val difference to the 16-student file.
# Usage as in the file: bash watch_growth.sh <KEY> <first seed> <last seed>. watch_growth2.sh of the same lab
# folder called it as: bash watch_growth.sh FD32P20 959 966   (FD32P20 = 32 wide students + 4 popularity-light).
# Its output is the source of the ledger block of 2026-11-16 03:26 UTC.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/kd/fd/watch_growth.sh
#
# watch_growth.sh <KEY> <first seed> <last seed>: wait for the fine-tuned test predictions of the new all-data wide students (both arms, the seed range), then run the stage-F acceptance,
# build + gate the file <KEY> (fd_finish.sh) and print the paired read against FD16P20. CPU only. Rule G (ledger 2026-11-15 21:0xZ).
cd <repo> || exit 1; K=$1; a=$2; b=$3; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
n=$(( (b - a + 1) * 2 ))
until [ $(for s in $(seq $a $b); do for arm in kd05w192 kd05t25w192; do [ -f nn_x/w5/preds/nn_w5_fd10_${arm}_s${s}_n2cpv3s3_test.npy ] && [ -f nn_x/w5/preds/nn_w5_fd10_${arm}_s${s}_n2cpv3s3_val_oof.npy ] && echo x; done; done | wc -l) -ge $n ]; do sleep 300; done
sleep 60; echo "=== $(date -u +%FT%TZ) $K: all $n new members fine-tuned ==="
OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/fulldata/accept_twins.py check F 2>&1 | tail -n 25 | cut -c1-260
bash nn_x/w12/kd/fd/fd_finish.sh $K 2>&1 | tail -n 14 | cut -c1-400
.venv/bin/python nn_x/w12/kd/build_r4.py FD16P20,$K 2>&1 | grep -E "^PAIR|final val" | cut -c1-520
