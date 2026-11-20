#!/usr/bin/env bash
# Launches one 10-epoch popularity-light member: the base recipe with the 8 popularity inputs excluded
# and the exposure count added. It checks the GPU (fewer than 4 processes, 60 GB free) before starting.
# Run as: bash launch_e10.sh <gpu> <seed>; seeds 951-954 at 07:11 UTC on 2026-11-13 (56.8 to 58.3
# minutes each); a copy under pop1/x8/ launched seeds 955-958 at 14:21 UTC (58 to 60 minutes each).
# Produced logs/abl_e10_pop1_s<seed>.{log,json}, the checkpoint and the predictions of each member.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/abl/pop1/launch_e10.sh
#
# launch ONE abl_e10_pop1 run: $1 = gpu, $2 = seed (stage-E command of ABLATION_PLAN_1113 §3.1 + the pop1 flags appended last)
set -uo pipefail
g=$1; s=$2; v=abl_e10_pop1_s$s
cd <repo>/nn_x/abl
n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
mem=$(nvidia-smi -i "$g" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits); used=${mem%%,*}; total=${mem##*,}
free=$(( (${total//[^0-9]/} - ${used//[^0-9]/}) / 1024 ))
echo "$(date -u +%FT%TZ) GPU $g: $n compute processes, $free GB free"
(( n < 4 && free >= 60 )) || { echo "GPU $g does not qualify: $v NOT launched"; exit 3; }
[[ -e logs/$v.log || -e ckpt/$v.pt ]] && { echo "$v outputs exist: NOT launched"; exit 4; }
POP=seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo
CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 \
setsid nohup <repo>/.venv/bin/python train.py --variant $v \
 --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout \
 --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2 \
 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500 \
 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed $s --xt_exclude $POP --xt_add xp_imp_c50 \
 > logs/$v.stdout 2>&1 < /dev/null &
sleep 3
p=""; for q in $(pgrep -f -- "train.py --variant $v "); do c=$(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null); [[ $c == "<repo>/.venv/bin/python train.py --variant $v "* ]] && p=$q; done
[[ -n $p ]] || { echo "could not resolve the train.py PID of $v"; exit 5; }
c=$(tr '\0' ' ' < /proc/$p/cmdline)
echo "$p train abl_e10 $v GPU $g ($(date -u +%FT%TZ); /proc/$p/cmdline verified: ${c:0:120}...) [lead pop1 10-ep family, pre-declared 07:10; E_BASE+HYG + --xt_exclude <8 POP> --xt_add xp_imp_c50; stop by this exact PID only]" >> RUNNING.txt
echo "LAUNCHED $v on GPU $g: PID $p"
