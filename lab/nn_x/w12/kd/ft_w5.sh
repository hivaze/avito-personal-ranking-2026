#!/usr/bin/env bash
# One member's 3-seed-bagged n2cpv3 fine-tune in the nn_x/w5 harness (same command as nn_x/abl/ftbag/ft_one_s3_r6.sh), then copy the
# FT and train-only prediction files into preds/ (cp --update=none). Usage: ft_w5.sh <gpu> <variant> [<variant> ...] (sequential on one GPU)
set -uo pipefail
g=${1:?gpu}; shift; R=<repo>; cd $R/nn_x/w5 || exit 1
export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
for v in "$@"; do
  [[ -f ckpt/$v.pt && -f preds/nn_${v}_trainonly_test.npy ]] || { echo "$(date -u +%FT%TZ) $v: ckpt or train-only test preds missing"; continue; }
  if [[ -f preds/nn_${v}_n2cpv3s3_val_oof.npy ]]; then echo "$(date -u +%FT%TZ) $v: FT exists, skipped"; else
    echo "$(date -u +%FT%TZ) START FT $v on GPU $g"
    $R/.venv/bin/python finetune.py --variant $v --out ${v}_n2cpv3s3 --extra n2c_pre_v3 --lr 5e-5 --epochs 6 --decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton --seeds 3 < /dev/null || { echo "FAILED FT $v"; continue; }
  fi
  for f in n2cpv3s3_val_oof n2cpv3s3_test trainonly_val trainonly_test; do cp --update=none preds/nn_${v}_${f}.npy $R/preds/; done
  echo "$(date -u +%FT%TZ) DONE FT $v"
done
