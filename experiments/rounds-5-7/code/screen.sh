#!/usr/bin/env bash
# Round 7: launcher of one run of the 2-epoch screen of block designs and never-varied hyper-parameters.
# Each arm is the baseline command plus the flags of that arm; the script also checks the GPU before launching.
# Run as (from the comment block below):
#   setsid nohup bash nn_x/r7/screen.sh 4 hc4 23 > nn_x/r7/logs/r7_hc4_s23.screen.out 2>&1 < /dev/null &
# 29 runs (13 arms and the baseline, seeds 23 and 24, one same-seed rerun) took 16.7 lane-hours on 2026-11-12.
# The model code with the arm flags is lab/nn_x/r7/model.py (1,040 lines, not copied here).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/r7/screen.sh
# r7 launcher for ONE run of the pre-declared r7 screen (reports/r7_build.md, reports/r7_research.md §5-6).
#
#   bash nn_x/r7/screen.sh <gpu> <arm> <seed>          # stage 1: 2-epoch screen config + the arm's flags
#   bash nn_x/r7/screen.sh <gpu> <arm> <seed> e10      # stage 2: 10-epoch F7 recipe + the arm's flags
#   bash nn_x/r7/screen.sh flags <arm> [e10]           # print the full train.py argument list only (no GPU, no process)
#   bash nn_x/r7/screen.sh check <gpu> [<arm>]         # GPU preflight only (process count + memory headroom)
#
# <arm>: base | muon_in | batch2s | resdrop0 | pbld | hc4 | dwa | diff2 | swiglu | derf | hl2 | periln | ftok | postln
#        r6-owned, ported from nn_x/r6 as of 04:20 UTC (a4u WITHOUT r6's later --retr_tcnt amendment): a4u | wd01 | cset |
#        typed | adrop0 | a4w. NOT part of the r7 batch: r6's own queue launched them at 04:51 UTC (nn_x/r6/logs).
#        drop0 (= --drop 0, all dropout sites; the pre-declared stage-2 combination if resdrop0 and adrop0 both promote)
#        combinations: arms joined by '+', e.g. muon_in+pbld (flags concatenated; batch2s may appear once).
# R7_SMOKE=1: plumbing smoke test (variant r7_smoke-<arm>_s<seed>, 20K train feeds, 1 epoch); never for the screen.
# R7_SUFFIX=r: appends to the variant name (A/A rerun: R7_SUFFIX=r bash screen.sh <gpu> base 23 -> r7_base_s23r).
# Variant: r7_<arm>_s<seed> (stage 1) / r7_<arm>_e10_s<seed> (stage 2); '+' in the arm becomes '-'.
# Outputs (all inside nn_x/r7): ckpt/<v>.pt, logs/<v>.{log,json,stdout}, preds/nn_<v>_trainonly_val.npy,
#   preds/holdout/nn_<v>_holdout.npy (written by train.py --save_holdout; holdout_eval.py is the fallback),
#   stage 2 also preds/nn_<v>_trainonly_test.npy.
#
# GPU rules (CURRENT_STATUS §3, task rules): before launching, the target GPU must have <= 3 compute processes (so <= 4
# with this one) and free memory >= NEED_GB(arm) + 80 GB headroom. Otherwise the script refuses (exit 3), or with WAIT=1
# polls every 180 s until the GPU qualifies. Check + launch + data load run under a per-GPU flock, so two r7 launches never
# load the train split on one GPU at the same moment. PIDs go to nn_x/r7/logs/RUNNING.txt; stop jobs by PID only.
# Launch detached, e.g.:
#   setsid nohup bash nn_x/r7/screen.sh 4 hc4 23 > nn_x/r7/logs/r7_hc4_s23.screen.out 2>&1 < /dev/null &
set -euo pipefail
R7=<repo>/nn_x/r7
PY=<repo>/.venv/bin/python

# stage 1 = the 2-epoch screen config: identical to nn_x/r6 BASE2 / nn_x/solvers sx_base (cvnd_pre2s, Muon lr 1e-3,
# WSD 0.2 / mix 0.1, retr16 + gate, K9 aux 0.5 -> 0.1, holdout-selected) + --no_test (no test preds needed) + --save_holdout
BASE2="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
--select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 \
--cooldown_mix 0.1 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 1000 --no_test --save_holdout"
# stage 2 = the F7 recipe (nn/run_final7b.sh, lr 2e-3, 10 epochs) + plateau snapshots + --save_holdout (test preds kept)
BASE10="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
--select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd \
--cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --lr 2e-3 --cooldown_mix 0.1 \
--save_steps 50000,60000,70000,77500 --save_holdout"

one_arm() {  # $1 = arm, $2 = "" or e10. Treatment flags are appended LAST (argparse: the last occurrence wins)
  case $1 in
    base) echo "" ;;
    muon_in) echo "--muon_inputs adamw" ;;
    # SDE square-root rule (Malladi 2022), pre-declared: batch x2 -> lr x sqrt2, (1-beta) x2 for every momentum,
    # warmup / eval interval / snapshot steps constant in samples (halved in steps)
    batch2s) if [[ $2 == e10 ]]; then
               echo "--cand_budget 65536 --max_feeds 1024 --lr 2.83e-3 --warmup 250 --eval_every 1250 --muon_momentum 0.90 --adam_betas 0.8,0.96 --save_steps 25000,30000,35000,38750"
             else
               echo "--cand_budget 65536 --max_feeds 1024 --lr 1.414e-3 --warmup 250 --eval_every 500 --muon_momentum 0.90 --adam_betas 0.8,0.96"
             fi ;;
    resdrop0) echo "--res_drop 0" ;;
    pbld) echo "--pbld 16" ;;
    hc4) echo "--hc 4" ;;
    dwa) echo "--dwa" ;;
    diff2) echo "--diff_attn cross" ;;
    swiglu) echo "--ffn swiglu" ;;
    derf) echo "--norm derf" ;;
    hl2) echo "--hist_extra 1" ;;
    periln) echo "--periln" ;;
    ftok) echo "--ftok 8" ;;
    postln) echo "--postln" ;;
    drop0) echo "--drop 0" ;;
    # r6-owned arms (nn_x/r6/queue_morning.sh arm_flags / arm_flags_e10, ported code)
    a4u) echo "--retr_uniq" ;;
    wd01) if [[ $2 == e10 ]]; then echo "--wd 0.05"; else echo "--wd 0.1"; fi ;;  # lr*wd = 1e-4 kept at 10 ep
    cset) echo "--cretr_m 8 --set_sim_bias" ;;
    typed) echo "--typed_cross" ;;
    adrop0) echo "--attn_drop 0" ;;
    a4w) echo "--retr_m 32 --retr_prod" ;;
    *) echo "screen.sh: unknown arm '$1'" >&2; return 1 ;;
  esac
}

arm_flags() {  # $1 = arm or arm+arm..., $2 = "" or e10
  local out="" a f
  IFS='+' read -r -a parts <<< "$1"
  for a in "${parts[@]}"; do f=$(one_arm "$a" "${2:-}") || return 1; out="$out $f"; done
  echo "$out"
}

need_gb() {  # GPU memory one run needs (nvidia-smi view). Measured (reports/r7_build.md §3.4): flag-off trainer peak 54.9 GB
  # reserved (+~2 GB CUDA context); batch2s + every other arm at once 61.5 GB; per-arm activation deltas <= +2 GB at 32K
  # candidates (tests/test_r7.py T9). Rounded up with margin:
  case $1 in
    *batch2s*) echo 72 ;;
    *hc4*|*ftok*|*a4w*) echo 66 ;;
    *) echo 62 ;;
  esac
}

gpu_ok() {  # $1 = gpu, $2 = need GB; prints the state, returns 0 if the run may start
  local g=$1 need=$2 n mem used total free
  n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  mem=$(nvidia-smi -i "$g" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  used=${mem%%,*}; total=${mem##*,}; used=${used//[^0-9]/}; total=${total//[^0-9]/}
  if [[ -z $used || -z $total ]]; then echo "GPU $g: cannot read memory with nvidia-smi" >&2; return 1; fi
  free=$(( (total - used) / 1024 ))
  echo "$(date -u +%FT%TZ) GPU $g: $n compute processes, ${free} GB free; this run needs ~${need} GB + 80 GB headroom, cap 4 processes"
  (( n <= 3 && free - need >= 80 ))
}

mode=${1:-}
case $mode in
  flags)
    f=$(arm_flags "$2" "${3:-}") || exit 1
    if [[ ${3:-} == e10 ]]; then echo "$BASE10 $f"; else echo "$BASE2 $f"; fi
    exit 0 ;;
  check)
    gpu_ok "$2" "$(need_gb "${3:-base}")" && echo "OK" || { echo "NOT OK"; exit 3; }
    exit 0 ;;
esac

gpu=$1; arm=$2; seed=$3; stage=${4:-}
case $gpu in 0|1|2|3|4|5|6|7) ;; *) echo "screen.sh: GPU must be one id 0-7" >&2; exit 1 ;; esac
[[ $stage == "" || $stage == e10 ]] || { echo "screen.sh: 4th argument must be empty or e10" >&2; exit 1; }
flags=$(arm_flags "$arm" "$stage")
tag=${arm//+/-}
if [[ ${R7_SMOKE:-0} == 1 ]]; then  # plumbing smoke test only: 20K train feeds, 1 epoch, evals every 25 steps
  flags="$flags --train_feeds 20000 --epochs 1 --eval_every 25"; tag=smoke-$tag
fi
if [[ $stage == e10 ]]; then v=r7_${tag}_e10_s${seed}; base=$BASE10; else v=r7_${tag}_s${seed}; base=$BASE2; fi
v=$v${R7_SUFFIX:-}  # e.g. R7_SUFFIX=r -> r7_base_s23r, the same-seed A/A rerun read by eval_screen.py aa
cd "$R7"
mkdir -p logs/locks preds/holdout
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
    POLARS_MAX_THREADS=12
if [[ -f logs/$v.json && -f preds/holdout/nn_${v}_holdout.npy ]]; then echo "$v already done"; exit 0; fi
need=$(need_gb "$arm")
lock=logs/locks/gpu${gpu}.lock
exec 9>"$lock"
flock 9  # serialise check + launch + data load per GPU (simultaneous train-split loads OOM'd a GPU before)
until gpu_ok "$gpu" "$need"; do
  if [[ ${WAIT:-0} != 1 ]]; then echo "screen.sh: GPU $gpu does not qualify; refusing (set WAIT=1 to poll)" >&2; exit 3; fi
  sleep 180
done
rc=0
if [[ ! -f logs/$v.json ]]; then  # train.py writes the json LAST
  echo "$(date -u +%FT%TZ) START $v on GPU $gpu: $flags"
  # shellcheck disable=SC2086
  $PY train.py --variant "$v" $base --seed "$seed" $flags > "logs/$v.stdout" 2>&1 &
  pid=$!
  echo "$(date -u +%FT%TZ) START pid=$pid screen_pid=$$ variant=$v gpu=$gpu arm=$arm seed=$seed stage=${stage:-s1}" >> logs/RUNNING.txt
  # hold the lock until the data is loaded and the model built (train.py logs "steps/epoch") or the job died
  until grep -q "steps/epoch" "logs/$v.stdout" 2>/dev/null || ! kill -0 "$pid" 2>/dev/null; do sleep 15; done
  flock -u 9
  wait "$pid" || rc=$?
  echo "$(date -u +%FT%TZ) END pid=$pid variant=$v rc=$rc" >> logs/RUNNING.txt
  (( rc == 0 )) || { echo "FAILED train $v (exit $rc)"; exit 1; }
else
  flock -u 9
fi
if [[ ! -f preds/holdout/nn_${v}_holdout.npy ]]; then  # fallback: holdout_eval.py loads the train split again (same lock)
  flock 9
  $PY holdout_eval.py "$v" > "logs/${v}_holdout.stdout" 2>&1 || rc=$?
  flock -u 9
  (( rc == 0 )) || { echo "FAILED holdout_eval $v (exit $rc)"; exit 1; }
fi
exec 9>&-
echo "$(date -u +%FT%TZ) DONE $v"
