#!/usr/bin/env bash
# FULL-DATA TWINS lane (reports/STALE_1115/fulldata.md; owner 2026-11-15: every submitted solution must be trained on all data).
# Same structure and the same GPU guard as nn_x/w5/lane.sh; runs a shared job list sequentially on ONE GPU (several lanes
# given the same list split it through atomic mkdir claims). NOT launched by its author: the lead launches.
#
#   bash lane_full.sh <gpu> "<job> <job> ..."       job = <stage>:<arm>:<seed>
#   DRY=1 bash lane_full.sh <gpu> "<jobs>"          print the commands only (no GPU check, no process, nothing written)
#   FD_SMOKE=1 ...                                  plumbing smoke: variant *_fdsmoke_*, 20K train feeds, 1 epoch, evals every
#                                                   25 steps (the W5_SMOKE settings of lane.sh)
# Stages (every command = the member's ORIGINAL command + "--holdout_in_train", run through the copies of this directory):
#   F:<arm>:<seed>   FULL RETRAIN twin of a W16 member. arm kd05w192 | kd05t25w192, seed 951-958 (also kd08w192).
#                    = lane.sh E:<arm>:<seed> (E_BASE + HYG + KD flags, 10 epochs, same --seed) -> variant w5_fd10_<arm>_s<seed>
#   R:<arm>:<seed>   RE-COOL twin (cooldown phase only, from the stored plateau snapshot w5_e10_<arm>_s<seed>@77500.pt,
#                    which was trained WITHOUT the holdout feeds) = lane.sh K:ref / L:ref (B_BASE + HYG, --seed 1000 + seed)
#                    -> variant w5_fdcd_<arm>_s<seed>
#   RH:<arm>:<seed>  R + --hzcd (the re-opened popularity-parity cooldown, lane.sh K:hz / L:hz) -> w5_fdcdhz_<arm>_s<seed>
#   FH:<arm>:<seed>  F + --hzcd: full 10-epoch retrain whose cooldown (last 20% of the steps) reads the hz popularity view
#                    (lane.sh E:hzcd form + the member's KD flags) -> w5_fd10hz_<arm>_s<seed>. The honest full-data twin of an
#                    hz member; NOT the same object as snapshot + hz re-cool (no optimiser reset at the cooldown start).
#   P:<arm>:<seed>   FULL RETRAIN twin of a popularity-light student (nn_x/abl). arm pop1kd05t25 (d128, teacher_mean25) |
#                    pop1kd05w192 (d192, teacher_mean11), seed 951-954 = nn_x/abl/pop1/launch_e10.sh + KD flags
#                    (nn_x/w12/kd/launch_pop1w.sh) -> variant abl_fd10_<arm>_s<seed>
#   PR:<arm>:<seed>  RE-COOL twin of a popularity-light student from abl_e10_<arm>_s<seed>@77500.pt -> abl_fdcd_<arm>_s<seed>
# Outputs go where the member's harness writes them (common.py of the code dir): nn_x/w5/{ckpt,logs,preds} for F / R / RH,
#   nn_x/abl/{ckpt,logs,preds} for P / PR. The twin names share no prefix with any existing family (w5_fd*, abl_fd*).
#   Holdout predictions of a twin are IN-SAMPLE and are written ONLY as preds/holdout/nn_<v>[@last]_holdoutINSAMPLE.npy.
#   NEVER run holdout_eval.py on a twin (it would write an in-sample file under the out-of-sample name); this lane has no
#   such fallback.
# GPU rules (= lane.sh): only GPUs in W5_GPUS (default "<ids>") are accepted; before EACH launch the GPU must have <= 3
#   compute processes and >= LANE_GB (60) GB free, checked with nvidia-smi under the per-GPU flock
#   nn_x/w5/logs/locks/gpu<g>.lock (the SAME lock file as lane.sh, for both harnesses), held until the job has loaded its
#   data (the log shows "steps/epoch"). A GPU that does not qualify is polled every 120 s (WAIT_MAX_H hours, default 6, then
#   the lane stops). CUDA_VISIBLE_DEVICES = exactly that id; CPU threads 12. PIDs go to <code dir>/logs/RUNNING_full.txt and
#   to this directory's logs/RUNNING_full.txt; stop jobs by exact PID only. Finished jobs (logs/<v>.json + the kept and
#   @last in-sample holdout files) are never re-run; a failed job leaves logs/queue/failed_<v>.
set -uo pipefail
R=<repo>; W5=$R/nn_x/w5; ABL=$R/nn_x/abl; FD=$R/nn_x/w12/stale/fulldata
PY=$R/.venv/bin/python; KD=$R/nn_x/w12/kd/out
gpu=${1:?gpu}; jobs=${2:?"job list <stage>:<arm>:<seed> ..."}; DRY=${DRY:-0}
ALLOWED=${W5_GPUS:-"0 1 2 4"}
[[ " $ALLOWED " == *" $gpu "* ]] || { echo "lane_full.sh: refusing GPU $gpu (allowed: $ALLOWED)" >&2; exit 1; }
# ---- copied verbatim from nn_x/w5/lane.sh (frozen copy ref/lane.sh.txt): E_BASE, B_BASE, HYG
B_BASE="--epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise \
--aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.02 --loss_norm feed --holdout 30000 --select_on holdout --eval_every 1000"
HYG="--save_last --save_holdout --select_min_step 12000 --cliplog 50"
E_BASE="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
--select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd \
--cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500"
# ---- nn_x/abl/pop1/launch_e10.sh: the popularity-light input flags
POP=seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo
FULL="--holdout_in_train"
kd_flags() {  # $1 = arm -> the member's KD flags, exactly as lane.sh arm_flags E:<arm> / launch_pop1w.sh
  case $1 in
    kd05w192)     echo "--kd_file $KD/teacher_mean11.npy --kd_w 0.5 --d 192" ;;
    kd05t25w192)  echo "--kd_file $KD/teacher_mean25.npy --kd_w 0.5 --d 192" ;;
    kd08w192)     echo "--kd_file $KD/teacher_mean11.npy --kd_w 0.8 --d 192" ;;
    pop1kd05t25)  echo "--kd_file $KD/teacher_mean25.npy --kd_w 0.5" ;;
    pop1kd05w192) echo "--kd_file $KD/teacher_mean11.npy --kd_w 0.5 --d 192" ;;
    *) echo "lane_full.sh: unknown arm $1" >&2; return 1 ;;
  esac
}
cd_flags() {  # re-cool form: the architecture comes from the snapshot cfg, so only the teacher file and kd_w are passed
  case $1 in
    kd05w192|pop1kd05w192)    echo "--kd_file $KD/teacher_mean11.npy --kd_w 0.5" ;;
    kd05t25w192|pop1kd05t25)  echo "--kd_file $KD/teacher_mean25.npy --kd_w 0.5" ;;
    kd08w192)                 echo "--kd_file $KD/teacher_mean11.npy --kd_w 0.8" ;;
    *) echo "lane_full.sh: unknown arm $1" >&2; return 1 ;;
  esac
}
check_gpu() {  # $1 = gpu; 0 iff <= 3 compute processes and >= LANE_GB GB free   (verbatim from lane.sh)
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
  [[ -z ${y:-} ]] || { echo "lane_full.sh: unexpected 4th field in job $j" >&2; exit 1; }
  case $st in
    F|R|RH|FH) H=$W5; TF=$FD/train_full.py; pre=w5
       [[ $arm == kd05w192 || $arm == kd05t25w192 || $arm == kd08w192 ]] || { echo "lane_full.sh: stage $st arms are kd05w192 | kd05t25w192 | kd08w192 (got $arm)" >&2; exit 1; }
       (( x >= 951 && x <= 958 )) || { echo "lane_full.sh: stage $st seeds are 951-958 (got $x)" >&2; exit 1; } ;;
    P|PR) H=$ABL; TF=$FD/train_full_abl.py; pre=abl
       [[ $arm == pop1kd05t25 || $arm == pop1kd05w192 ]] || { echo "lane_full.sh: stage $st arms are pop1kd05t25 | pop1kd05w192 (got $arm)" >&2; exit 1; }
       (( x >= 951 && x <= 954 )) || { echo "lane_full.sh: stage $st seeds are 951-954 (got $x)" >&2; exit 1; } ;;
    *) echo "lane_full.sh: unknown stage $st (F | FH | R | RH | P | PR)" >&2; exit 1 ;;
  esac
  case $st in
    F)  f=$(kd_flags "$arm") || exit 1
        v=${pre}_fd10_${arm}_s${x}; args="$E_BASE $HYG --seed $x $f $FULL" ;;
    FH) f=$(kd_flags "$arm") || exit 1
        v=${pre}_fd10hz_${arm}_s${x}; args="$E_BASE $HYG --seed $x $f --hzcd $FULL" ;;
    P)  f=$(kd_flags "$arm") || exit 1
        v=${pre}_fd10_${arm}_s${x}; args="$E_BASE $HYG --seed $x --xt_exclude $POP --xt_add xp_imp_c50 $f $FULL" ;;
    R|RH|PR) f=$(cd_flags "$arm") || exit 1
        init=$H/ckpt/${pre}_e10_${arm}_s${x}@77500.pt
        [[ -f $init ]] || { echo "lane_full.sh: missing snapshot $init" >&2; exit 1; }
        v=${pre}_fdcd_${arm}_s${x}; hz=""
        [[ $st == RH ]] && { v=${pre}_fdcdhz_${arm}_s${x}; hz=" --hzcd"; }
        args="--init $init $B_BASE $HYG --seed $((1000 + x)) $f$hz $FULL" ;;
  esac
  if [[ ${FD_SMOKE:-0} == 1 ]]; then
    v=${v/_fd/_fdsmoke_}; args="$args --train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25 --cliplog 10"
  fi
  cmd="$PY $TF --variant $v $args"
  if [[ $DRY == 1 ]]; then echo "GPU $gpu: (cd $H && PYTHONPATH=$H) $cmd"; continue; fi
  cd "$H" || exit 1
  mkdir -p logs/queue preds/holdout ckpt "$W5/logs/locks" "$FD/logs"
  export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
      POLARS_MAX_THREADS=12 PYTHONPATH=$H PYTHONDONTWRITEBYTECODE=1
  done_() { [[ -f logs/$v.json && -f preds/holdout/nn_${v}_holdoutINSAMPLE.npy && -f preds/holdout/nn_${v}@last_holdoutINSAMPLE.npy ]]; }
  if done_; then echo "$(date -u +%FT%TZ) $v already done"; continue; fi
  [[ -e logs/$v.log || -e ckpt/$v.pt ]] && { echo "$(date -u +%FT%TZ) $v: outputs of an unfinished run exist: NOT relaunched (inspect, then remove them by hand)"; continue; }
  mkdir "logs/queue/claim_$v" 2>/dev/null || continue   # claimed by another lane
  exec 9>"$LOCK"; flock 9
  t0=$(date +%s)
  until check_gpu "$gpu"; do
    (( $(date +%s) - t0 > ${WAIT_MAX_H:-6} * 3600 )) && { echo "$(date -u +%FT%TZ) GPU $gpu never qualified: lane stops"; \
      rmdir "logs/queue/claim_$v"; flock -u 9; exit 3; }
    sleep 120
  done
  echo "$(date -u +%FT%TZ) START $v on GPU $gpu: $cmd"
  $cmd > "logs/$v.stdout" 2>&1 < /dev/null &
  p=$!
  pc=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null || true)
  [[ $pc == "$PY $TF --variant $v "* ]] && ok=verified || ok=UNVERIFIED
  line="$(date -u +%FT%TZ) START pid=$p (/proc cmdline $ok) lane_pid=$$ variant=$v gpu=$gpu job=$j [FULL-DATA twin, reports/STALE_1115/fulldata.md; stop by this exact PID only]"
  echo "$line" >> logs/RUNNING_full.txt; echo "$line" >> "$FD/logs/RUNNING_full.txt"
  until grep -q "steps/epoch" "logs/$v.stdout" 2>/dev/null || ! kill -0 "$p" 2>/dev/null; do sleep 15; done
  flock -u 9; exec 9>&-
  rc=0; wait "$p" || rc=$?
  line="$(date -u +%FT%TZ) END pid=$p variant=$v rc=$rc"
  echo "$line" >> logs/RUNNING_full.txt; echo "$line" >> "$FD/logs/RUNNING_full.txt"
  if (( rc != 0 )); then echo "$(date -u +%FT%TZ) FAILED $v (rc $rc)"; touch "logs/queue/failed_$v"; continue; fi
  if ! done_; then echo "$(date -u +%FT%TZ) INCOMPLETE $v (rc 0 but an expected output is missing)"; touch "logs/queue/failed_$v"; continue; fi
  echo "$(date -u +%FT%TZ) DONE $v"
done
