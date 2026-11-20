#!/usr/bin/env bash
# Launcher of stage A: one click-label 10-epoch member with the three new input columns (arm nx3) or without
# them (arm base), same command otherwise.
# Run as: bash nn_x/w12/nx3/launch_nx3_hl.sh <gpu> <seed> nx3|base
# Eight members with the input (seeds 951-958) were compared with six stored and two fresh members without it.
# Stage B used lab/nn_x/w12/nx3/launch_nx3_kd.sh, the same for distilled width-192 students.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/nx3/launch_nx3_hl.sh
# STAGE A of the trained-input test nx3 (rule: ledger nn_x/w5/logs/E_kd_predeclared.txt, block "TRAINED-INPUT TEST nx3", amendment A1:
# bundle = nx_rp3_z, nx_rp3_rk, has:nx_rp3_z; design reports/INPUTS_1116/DESIGN.md, critic reports/INPUTS_1116/CRITIC.md).
# launch ONE hard-label 10-epoch d128 member in nn_x/abl: $1 = gpu, $2 = seed, $3 = nx3 | base
#   nx3  -> abl_e10_nx3_s<seed>   : E_BASE + HYG (the command of nn_x/abl/pop1/launch_e10.sh WITHOUT its two pop1 flags) + --xt_add <3 NX inputs>
#   base -> abl_e10_base_s<seed>  : the same command without --xt_add (fresh control in the same harness; seeds 957, 958)
# Write-new copy of reports/INPUTS_1116/design/launch_nx_hl.sh (nothing existing is edited): arms nx / nx4 replaced by nx3, every training flag
# unchanged; added: the per-GPU flock of the lanes (nn_x/w5/logs/locks/gpu<g>.lock, held from the GPU check until the job has loaded its data).
# Like every launcher of nn_x/abl it creates new files in
# nn_x/abl/{logs,ckpt,preds} and APPENDS one registry line to nn_x/abl/RUNNING.txt. GPU rule as gpu_ok.sh: < 4 compute processes, >= 60 GB free.
# SMOKE=1 -> variant abl_smoke_<arm>_s<seed>, 20,000 train feeds, 1 epoch (3-4 min).
set -uo pipefail
g=${1:?gpu}; s=${2:?seed}; arm=${3:?nx3|base}
NX3=nx_rp3_z,nx_rp3_rk,has:nx_rp3_z
case $arm in nx3) XA="--xt_add $NX3" ;; base) XA="" ;; *) echo "arm must be nx3 | base"; exit 1 ;; esac
v=abl_e10_${arm}_s$s; EXTRA=""
[[ ${SMOKE:-0} == 1 ]] && { v=abl_smoke_${arm}_s$s; EXTRA="--train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25 --cliplog 10"; }
cd <repo>/nn_x/abl
[[ -e logs/$v.log || -e logs/$v.stdout || -e ckpt/$v.pt ]] && { echo "$v outputs exist: NOT launched"; exit 4; }
if [[ $arm != base ]]; then
  for n in $(echo "${XA#--xt_add }" | tr ',' ' '); do f=${n#has:}; for sp in train val test; do
    [[ -f <repo>/nn/feat_cache/$sp/$f.npy ]] || { echo "missing nn/feat_cache/$sp/$f.npy: NOT launched"; exit 6; }; done; done; fi
exec 9><repo>/nn_x/w5/logs/locks/gpu${g}.lock; flock 9   # per-GPU lock of the lanes: check + launch under the lock
t0=$(date +%s); until bash ftbag/gpu_ok.sh $g; do (( $(date +%s) - t0 > 14400 )) && { echo "GPU $g never qualified"; flock -u 9; exit 3; }; sleep 60; done
CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 \
setsid nohup <repo>/.venv/bin/python train.py --variant $v \
 --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout \
 --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2 \
 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500 \
 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed $s $XA $EXTRA \
 > logs/$v.stdout 2>&1 < /dev/null 9>&- &
sleep 5
p=""; for q in $(pgrep -f -- "train.py --variant $v "); do c=$(tr '\0' ' ' < /proc/$q/cmdline 2>/dev/null); [[ $c == "<repo>/.venv/bin/python train.py --variant $v "* ]] && p=$q; done
[[ -n $p ]] || { echo "could not resolve the train.py PID of $v"; flock -u 9; exit 5; }
# hold the lock until the job is visible in nvidia-smi AND has loaded its data ("steps/epoch" in its log, as nn_x/w5/lane.sh), or has ended (never longer than 30 min)
tl=$(date +%s); until { nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -qx "$p" && grep -q "steps/epoch" logs/$v.stdout 2>/dev/null; } || ! kill -0 $p 2>/dev/null || (( $(date +%s) - tl > 1800 )); do sleep 5; done
flock -u 9; exec 9>&-
echo "$p train abl_e10 $v GPU $g ($(date -u +%FT%TZ); /proc/$p/cmdline verified) [trained-input test nx3, stage A hard labels; ledger block TRAINED-INPUT TEST nx3; stop by this exact PID only]" >> RUNNING.txt
echo "LAUNCHED $v on GPU $g: PID $p"
