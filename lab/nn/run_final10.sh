#!/usr/bin/env bash
# nn_final10 / nn_final11 (pre-declared 23:05 UTC): ARCHITECTURE-diverse 10-epoch families for the ensemble (strength-diversity law):
# --arch gru (F10, variants v23_gru_pre2s_s5xx) or --arch din (F11, v24_din_pre2s_s6xx); cvnd_pre2s + recipe (ii)
# (Muon lr 2e-3, WSD 0.2, mix 0.1) + retr16 + gate; holdout-selected; n2cpv3 fta + holdout_eval. Smoke-tested 23:00 (491 steps OK).
# Usage: bash nn/run_final10.sh <gpu> <arch> <seed> "<bash condition to wait for>"
set -euo pipefail
cd "$(dirname "$0")"
gpu=$1; arch=$2; seed=$3; cond=${4:-true}
until eval "$cond"; do sleep 60; done
if [[ $arch == gru ]]; then v=v23_gru_pre2s_s$seed; else v=v24_din_pre2s_s$seed; fi
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16
[[ -f ckpt/$v.pt ]] || ../.venv/bin/python train.py --variant $v --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 \
    --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 \
    --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate \
    --lr 2e-3 --cooldown_mix 0.1 --seed $seed --arch $arch
EXTRA=n2c_pre_v3 SUF=n2cpv3 FTARGS='--decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton' bash ft_list.sh $v
[[ -f ../preds/holdout/nn_${v}_holdout.npy ]] || ../.venv/bin/python holdout_eval.py $v
