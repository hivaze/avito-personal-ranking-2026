#!/usr/bin/env bash
# ftarm_launch.sh (2026-11-16, reports/NEXT_1116/fillers, lane B): ONE launcher for the paired fine-tune arms (rule text: docstring of ftarm_read.py). NOT run by lane B: the lead launches after the critic.
#   ftarm_launch.sh "<kinds>" [n lanes = 4]          e.g.  ftarm_launch.sh "bs05 bs2 xm30" 4
#   DRY=1 ftarm_launch.sh "<kinds>"                   print the declaration and every fine-tune command; nothing is written, nothing runs
# Steps: (0) refuse if a declaration exists (one declaration, one family: the Holm family is fixed here, before the first run);
#        (1) write ftarm_DECLARED.txt (arms, k, UTC time, the rule line);
#        (2) PATH CHECK OF THE DAY: re-run ONE control (member w5_fd10_kd05w192_s951, suffix ftctl3c, the command of the stored controls) and run ftarm_pathcheck.py
#            -> logs/ftarm_pathcheck_out.txt; anything but "PATH CHECK: PASS" stops here, no arm is launched;
#        (3) n lanes of ftarm_lane.sh, every lane given the SAME job list (arm-major: all 8 members of the first kind, then the next kind); the lanes split it through atomic claims;
#        (4) wait for the lanes; print the command of the ONE read (ftarm_read.py). The read itself is the lead's (one look, after the ledger entry).
# Every fine-tune goes through the GPU guard of ftarm_lane.sh (<= 3 compute processes and >= 60 GB free on the GPU, per-GPU flock taken non-blocking). Nothing is ever signalled.
# Stop: touch reports/NEXT_1116/fillers/logs/FTARM_STOP before step 3 and the arms are not launched. Once the lanes run, a fine-tune is stopped only by its owner, by the exact PID
#   recorded in ftarm_RUNNING.txt (after reading /proc/<pid>/cmdline); a lane whose job failed or was stopped does not start its remaining jobs.
set -uo pipefail
kinds=${1:?"kinds, e.g. \"bs05 bs2 xm30\""}; NL=${2:-4}; DRY=${DRY:-0}
R=<repo>; F=$R/reports/NEXT_1116/fillers; DEC=$F/ftarm_DECLARED.txt; STOP=$F/logs/FTARM_STOP
MEM=""; for a in kd05w192 kd05t25w192; do for s in 951 952 953 954; do MEM="$MEM w5_fd10_${a}_s$s"; done; done
for k in $kinds; do case $k in bs05|bs2|xm30|xm3|lr25|lr100|wd0|wd10) ;; *) echo "unknown kind $k" >&2; exit 1 ;; esac; done
nk=$(echo $kinds | wc -w)
JOBS=""; for k in $kinds; do for v in $MEM; do JOBS="$JOBS $v:ft$k:$k"; done; done
RULE="PASS(arm) iff FILE-level arm minus control (mixctl3): val >= +2 SE with Holm over the $nk declared arm(s) AND pmix-private >= 0 AND val M<50 >= -1 SE; a PASS only buys a 3-seed replication on the 16 all-data members, then a candidate for PICK 2, never pick 1; else CLOSED."
if [[ $DRY == 1 ]]; then
  echo "DRY: declaration that would be written to $DEC:"; echo "arms: $kinds"; echo "k: $nk"; echo "rule: $RULE"
  echo "DRY: path-check control:"; DRY=1 bash $F/ftarm_lane.sh dry w5_fd10_kd05w192_s951:ftctl3c:ctl
  echo "DRY: $(echo $JOBS | wc -w) arm fine-tunes on $NL lanes:"; DRY=1 bash $F/ftarm_lane.sh dry $JOBS
  exit 0
fi
[[ -e $DEC ]] && { echo "$DEC exists: arms were already declared (one declaration only)" >&2; exit 1; }
mkdir -p $F/logs
{ echo "arms: $kinds"; echo "k: $nk"; echo "declared: $(date -u +%FT%TZ)"; echo "control suffix: mixctl3 (stored controls of Q4 MIXFT; admitted only by logs/ftarm_pathcheck_out.txt = PATH CHECK: PASS)"; echo "rule: $RULE"; } > $DEC
echo "$(date -u +%FT%TZ) declared: $(tr '\n' ';' < $DEC)"
bash $F/ftarm_lane.sh pc w5_fd10_kd05w192_s951:ftctl3c:ctl > $F/logs/lane_pc.log 2>&1
[[ -f $R/nn_x/w5/preds/nn_w5_fd10_kd05w192_s951_ftctl3c_val_oof.npy ]] || { echo "$(date -u +%FT%TZ) path-check control did not finish: stop (see $F/logs/lane_pc.log)"; exit 2; }
( cd $R && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 .venv/bin/python $F/ftarm_pathcheck.py ftctl3c ) > $F/logs/ftarm_pathcheck_out.txt 2>&1
tail -n 3 $F/logs/ftarm_pathcheck_out.txt
grep -q "^PATH CHECK: PASS" $F/logs/ftarm_pathcheck_out.txt || { echo "$(date -u +%FT%TZ) PATH CHECK did not pass: no arm is launched"; exit 3; }
[[ -e $STOP ]] && { echo "$(date -u +%FT%TZ) stop file present: arms not launched"; exit 4; }
for i in $(seq 1 $NL); do
  setsid nohup bash $F/ftarm_lane.sh a$i $JOBS > $F/logs/lane_a$i.log 2>&1 < /dev/null &
  sleep 2
done
echo "$(date -u +%FT%TZ) $NL lanes started for $(echo $JOBS | wc -w) fine-tunes; PIDs of the fine-tunes are appended to $F/ftarm_RUNNING.txt"
until [[ $(ls $F/logs/lane_a*.done 2>/dev/null | wc -l) -ge $NL ]]; do sleep 60; done   # every lane ends by touching its .done file (a failed lane also leaves .failed)
echo "$(date -u +%FT%TZ) all lanes ended ($(ls $F/logs/lane_a*.failed 2>/dev/null | wc -l) failed). READ (one look, the lead's): cd $R && CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=12 .venv/bin/python reports/NEXT_1116/fillers/ftarm_read.py > reports/NEXT_1116/fillers/out/ftarm_read_out.txt"
