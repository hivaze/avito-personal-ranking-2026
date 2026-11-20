#!/usr/bin/env bash
# fill.sh <jobs file> [MAXP=2]: low-priority filler for re-cool arms. Every 60 s, for each GPU: if the GPU has <= MAXP compute processes and >= 60 GB free, launch ONE single-job
# lane (nn_x/w5/lane.sh <gpu> <job>) for the next job of the list that is neither finished, claimed nor failed. A lane is only started when its slot is free, so no filler lane ever
# WAITS holding a per-GPU lock: fine-tune lanes and the full-data lanes (which need a GPU with <= 3 processes) always find their slot (with MAXP = 2 one slot per GPU stays spare).
# At most one launch per GPU per 150 s (the new process must be visible before the next count). Stop: touch nn_x/w12/kd/logs/FILL_STOP (running jobs are not touched).
R=<repo>; W=$R/nn_x/w5; J=${1:?jobs file}; MAXP=${2:-2}; S=$R/nn_x/w12/kd/logs/FILL_STOP; declare -A last
vname() { IFS=: read -r st arm x <<< "$1"; case $st in K) echo w5_kcd_${arm}_s$x ;; L) echo w5_lcd_${arm}_s$x ;; E) echo w5_e10_${arm}_s$x ;; *) echo "" ;; esac; }
while [ ! -e $S ]; do
  pend=0
  for g in <ids>; do
    [ -e $S ] && break
    now=$(date +%s); (( now - ${last[$g]:-0} < 150 )) && { pend=1; continue; }
    job=""; for j in $(cat $J); do v=$(vname $j); [ -z "$v" ] && continue
      [[ -f $W/logs/$v.json || -d $W/logs/queue/claim_$v || -e $W/logs/queue/failed_$v || -e $W/logs/$v.log || -e $W/ckpt/$v.pt ]] && continue; job=$j; break; done
    [ -z "$job" ] && break
    pend=1
    n=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true); fr=$(nvidia-smi -i $g --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [[ -n $fr && $n -le $MAXP && $fr -ge 61440 ]] || continue
    echo "$(date -u +%FT%TZ) GPU $g ($n processes, $fr MiB free): launch $job"
    (cd $W && W5_GPUS="<ids>" WAIT_MAX_H=1 setsid nohup bash lane.sh $g "$job" >> logs/fill_$(vname $job).log 2>&1 < /dev/null &)
    last[$g]=$(date +%s); sleep 3
  done
  [ $pend -eq 0 ] && { echo "$(date -u +%FT%TZ) job list exhausted"; break; }
  sleep 60
done
echo "$(date -u +%FT%TZ) fill.sh ends"
