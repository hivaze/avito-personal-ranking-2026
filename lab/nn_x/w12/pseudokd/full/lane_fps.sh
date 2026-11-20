#!/usr/bin/env bash
# ALL-DATA PSEUDO-FEED STUDENTS lane (2026-11-17 ~22:5xZ, lead; owner after the PSEUDO-KD stage-2 NO PASS asked for the files to be built anyway
# ). Owner-requested PROBE files, not pick candidates: picks change only by the docs/11 rules.
# = lab/nn_x/w12/stale/fulldata/lane_full.sh stage F (the all-data twins behind FD16P20: E_BASE + HYG + --seed + the member's KD flags +
#   --holdout_in_train, 10 epochs) + the pseudo-feed flags of lane_pk.sh E:kd05w192ps, run with the trainer
#   lab/nn_x/w12/pseudokd/train_full_pk.py (= pseudokd/train.py + the full-data patch of train_full4.py, port check in logs/port_check.txt).
#   bash lane_fps.sh <gpu> "F:<arm>:<seed> ..."   arm kd05w192 | kd05t25w192, seed 951-958 -> variant w5_fdxps_<arm>_s<seed>
#                                                (build_r4.py keys FDX16P20:ps = both arms replaced; FDXH16P20:ps = only kd05w192 replaced)
#   DRY=1 ...       print the commands only;   FPS_SMOKE=1 ...  20K train feeds, 1 epoch, evals every 25 steps -> variant w5_fdxpssmoke_*
# GPU rule (= lane.sh / lane_full.sh, verbatim check_gpu): before EACH launch <= 3 compute processes and >= 60 GB free under the per-GPU flock
#   nn_x/w5/logs/locks/gpu<g>.lock, held until the job logs "steps/epoch"; CUDA_VISIBLE_DEVICES = that id; CPU threads 12.
# Holdout predictions are IN-SAMPLE (..._holdoutINSAMPLE.npy only). PIDs in logs/RUNNING.txt; stop by exact PID only.
set -uo pipefail
R=<repo>; W5=$R/nn_x/w5; KD=$R/nn_x/w12/kd/out; D=$R/lab/nn_x/w12/pseudokd/full
TF=$R/lab/nn_x/w12/pseudokd/train_full_pk.py; PY=$R/.venv/bin/python
gpu=${1:?gpu}; jobs=${2:?"job list F:<arm>:<seed> ..."}; DRY=${DRY:-0}
[[ " <ids> " == *" $gpu "* ]] || { echo "lane_fps.sh: refusing GPU $gpu" >&2; exit 1; }
# ---- verbatim from lane_full.sh / nn_x/w5/lane.sh
HYG="--save_last --save_holdout --select_min_step 12000 --cliplog 50"
E_BASE="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
--select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd \
--cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500"
FULL="--holdout_in_train"
# ---- verbatim from lane_pk.sh
PSEUDO="--pseudo_ratio 1.0 --pseudo_w 0.3 --pseudo_head 1 --pseudo_gap_min 16 --pseudo_gap_max 64 --pseudo_tmax 63"
kd_flags() {  # verbatim from lane_full.sh kd_flags
  case $1 in
    kd05w192)     echo "--kd_file $KD/teacher_mean11.npy --kd_w 0.5 --d 192" ;;
    kd05t25w192)  echo "--kd_file $KD/teacher_mean25.npy --kd_w 0.5 --d 192" ;;
    *) echo "lane_fps.sh: unknown arm $1" >&2; return 1 ;;
  esac
}
check_gpu() {  # verbatim from lane.sh
  local n mem used total free need=${LANE_GB:-60}
  n=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  mem=$(nvidia-smi -i "$1" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  used=${mem%%,*}; total=${mem##*,}; used=${used//[^0-9]/}; total=${total//[^0-9]/}
  if [[ -z $used || -z $total ]]; then echo "$(date -u +%FT%TZ) GPU $1: cannot read nvidia-smi"; return 1; fi
  free=$(( (total - used) / 1024 ))
  echo "$(date -u +%FT%TZ) GPU $1: $n compute processes, ${free} GB free (need <= 3 and >= ${need} GB)"
  (( n <= 3 && free >= need ))
}
LOCK=$W5/logs/locks/gpu${gpu}.lock
for j in $jobs; do
  IFS=: read -r st arm x y <<< "$j"
  [[ $st == F && -z ${y:-} ]] || { echo "lane_fps.sh: job must be F:<arm>:<seed> (got $j)" >&2; exit 1; }
  f=$(kd_flags "$arm") || exit 1
  (( x >= 951 && x <= 958 )) || { echo "lane_fps.sh: seeds 951-958 (got $x)" >&2; exit 1; }
  v=w5_fdxps_${arm}_s${x}; args="$E_BASE $HYG --seed $x $f $PSEUDO $FULL"
  if [[ ${FPS_SMOKE:-0} == 1 ]]; then
    v=w5_fdxpssmoke_${arm}_s${x}; args="$args --train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25 --cliplog 10"
  fi
  cmd="$PY $TF --variant $v $args"
  if [[ $DRY == 1 ]]; then echo "GPU $gpu: (cd $W5 && PYTHONPATH=$W5:$R/nn_x/w12/kd/fd) $cmd"; continue; fi
  cd "$W5" || exit 1
  mkdir -p logs/queue preds/holdout ckpt logs/locks "$D/logs"
  export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
      POLARS_MAX_THREADS=12 PYTHONPATH=$W5:$R/nn_x/w12/kd/fd PYTHONDONTWRITEBYTECODE=1
  done_() { [[ -f logs/$v.json && -f preds/holdout/nn_${v}_holdoutINSAMPLE.npy && -f preds/holdout/nn_${v}@last_holdoutINSAMPLE.npy ]]; }
  if done_; then echo "$(date -u +%FT%TZ) $v already done"; continue; fi
  [[ -e logs/$v.log || -e ckpt/$v.pt ]] && { echo "$(date -u +%FT%TZ) $v: outputs of an unfinished run exist: NOT relaunched"; continue; }
  mkdir "logs/queue/claim_$v" 2>/dev/null || continue
  exec 9>>"$LOCK"; flock 9
  t0=$(date +%s)
  until check_gpu "$gpu"; do
    (( $(date +%s) - t0 > ${WAIT_MAX_H:-6} * 3600 )) && { echo "$(date -u +%FT%TZ) GPU $gpu never qualified: lane stops"; rmdir "logs/queue/claim_$v"; flock -u 9; exit 3; }
    sleep 120
  done
  echo "$(date -u +%FT%TZ) START $v on GPU $gpu: $cmd"
  $cmd > "logs/$v.stdout" 2>&1 < /dev/null &
  p=$!
  pc=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null || true)
  [[ $pc == "$PY $TF --variant $v "* ]] && ok=verified || ok=UNVERIFIED
  echo "$(date -u +%FT%TZ) START pid=$p (/proc cmdline $ok) lane_pid=$$ variant=$v gpu=$gpu job=$j [stop by this exact PID only]" >> "$D/logs/RUNNING.txt"
  until grep -q "steps/epoch" "logs/$v.stdout" 2>/dev/null || ! kill -0 "$p" 2>/dev/null; do sleep 15; done
  flock -u 9; exec 9>&-
  rc=0; wait "$p" || rc=$?
  echo "$(date -u +%FT%TZ) END pid=$p variant=$v rc=$rc" >> "$D/logs/RUNNING.txt"
  if (( rc != 0 )); then echo "$(date -u +%FT%TZ) FAILED $v (rc $rc)"; touch "$D/logs/failed_$v"; exit 4; fi
  if ! done_; then echo "$(date -u +%FT%TZ) INCOMPLETE $v"; touch "$D/logs/failed_$v"; exit 5; fi
  echo "$(date -u +%FT%TZ) DONE $v"
done
