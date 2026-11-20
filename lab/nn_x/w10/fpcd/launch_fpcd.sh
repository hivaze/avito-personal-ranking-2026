#!/usr/bin/env bash
# nn_x/w10/fpcd WAVE-10 FPCD-lite launcher (rule nn_x/w5/logs/B_fpcd_predeclared.txt, frozen 2026-11-13T16:53:48Z). Written 2026-11-13 ~17:00 UTC.
# Starts 2 lane.sh lanes on each of GPUs N and 7 (the FPCD agent's grant), staggered 45 s, all sharing ONE job list
# (atomic mkdir claims in logs/queue -> every job runs once): B:fpcd:101 B:fpcd:102 B:fpcd:103 B:fpcd:104.
# Each lane checks its GPU before the launch (<= 3 compute processes, >= 60 GB free; nvidia-smi under the per-GPU flock) and polls
# otherwise. Lane / python PIDs go to nn_x/w10/fpcd/RUNNING.txt (lane.sh START/END lines) and are mirrored to nn_x/w5/logs/RUNNING_w5.txt.
#   setsid nohup bash nn_x/w10/fpcd/launch_fpcd.sh > nn_x/w10/fpcd/logs/launch_fpcd.out 2>&1 < /dev/null &
set -uo pipefail
FP=<repo>/nn_x/w10/fpcd
RUN=$FP/RUNNING.txt
RUN5=<repo>/nn_x/w5/logs/RUNNING_w5.txt
J="B:fpcd:101 B:fpcd:102 B:fpcd:103 B:fpcd:104"
GPUS=${GPUS:-"<ids>"}; NL=${LANES_PER_GPU:-2}
echo "# $(date -u +%FT%TZ) [W10 FPCD launch] lane.sh x $NL per GPU on '$GPUS', shared J=\"$J\"" | tee -a "$RUN" >> "$RUN5"
for k in $(seq 1 "$NL"); do
  for g in $GPUS; do
    out=$FP/logs/laneB_gpu${g}_${k}.out
    setsid nohup bash "$FP/lane.sh" "$g" "$J" > "$out" 2>&1 < /dev/null &
    p=$!
    echo "$(date -u +%FT%TZ) lane wrapper pid=$p gpu=$g lane=$k jobs=\"$J\" log logs/laneB_gpu${g}_${k}.out [W10 FPCD launch]" | tee -a "$RUN" >> "$RUN5"
    sleep 45
  done
done
echo "$(date -u +%FT%TZ) [W10 FPCD launch] all lanes started"
