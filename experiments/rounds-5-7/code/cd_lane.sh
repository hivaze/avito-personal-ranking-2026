#!/usr/bin/env bash
# Round 6: the cooldown-continuation harness. It re-runs only the last 2 epochs (the cooldown) of a finished
# 10-epoch member from its stored checkpoint at step 77,500, with one change per arm: the share of random-order
# feeds mixed into the cooldown (m02 = 2%, m0 = none, c0 = the original 10%), weight decay, dropout, and others.
# One arm of this harness, the 2% mix, became the cooldown of every later member.
# Run as shown in the comment block below, for example:
#   setsid nohup bash nn_x/r6/cd_lane.sh 4 "c0:101 m02:101 w10:101 ..." > nn_x/r6/logs/lane_cd_gpu4_1.out 2>&1 &
# It wrote checkpoints, logs and predictions under nn_x/r6/{ckpt,logs,preds}.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/r6/cd_lane.sh
# R6_PLAN section 4, queue Q1: cooldown-continuation (CD) screen lane. Written 2026-11-12 ~05:00 UTC by the r6 plan step;
# DRY-tested only (no GPU job launched by its author). Re-runs F7's own cooldown (2 epochs = 19,670 steps, lr 2e-3 ->
# 3% floor, 1-sqrt) from the saved pre-cooldown snapshot nn/ckpt/v17_pre2s_s<root>@77500.pt (read-only), with one
# cooldown-phase change per arm. Outputs go to nn_x/r6/{ckpt,preds,logs} (common.py). Model flags are not allowed with
# --init (train.py:186), so only training-loop knobs appear here.
#
#   bash nn_x/r6/queue_morning.sh check "4"                                   # preflight (process / memory refusal)
#   J="c0:101 m02:101 w10:101 c0:102 m02:102 w10:102 c0:103 m02:103 w10:103 c0:104 m02:104 w10:104"
#   for k in 1 2; do setsid nohup bash nn_x/r6/cd_lane.sh 4 "$J" > nn_x/r6/logs/lane_cd_gpu4_$k.out 2>&1 &
#     echo "$! lane cd GPU N #$k ($(date -u +%FT%TZ))" >> nn_x/r6/logs/RUNNING_r6.txt; done
#   DRY=1 bash nn_x/r6/cd_lane.sh 4 "$J"                                      # print the commands only
#
# Lanes given the same job list split it through atomic mkdir claims; a finished job (logs/<v>.json + holdout preds)
# is never re-run. Two lanes of a GPU never load data at the same moment (same per-GPU flock as queue_morning.sh).
#
# Wave 2 (2026-11-12 ~08:45 UTC): arms d0 / d20 (--drop_cd), pp02 (--pool_p), aug (--aug_cand/--aug_hist), flags in
# train.py + cd_arms.py, tests/test_cdw2.py. LANES=1 (default 2) preflights ONE lane on the GPU (refuse at >= 4 compute
# processes after this lane, or < LANE_GB=60 GB free); the default keeps queue_morning.sh's 2-lane check unchanged.
#   J="d0:101 d20:101 pp02:101 aug:101 ..."; LANES=1 setsid nohup bash nn_x/r6/cd_lane.sh 0 "$J" > ... 2>&1 < /dev/null &
#
# Wave 4 (2026-11-12 ~16:00 UTC, workflow wx9m2kck8; RESEARCH_SWEEP_1112 P5/P4/P3; flags in train.py + cd_arms4.py,
# tests/test_cdw4.py; readout r6_eval_cdw4.py, reference m02): every wave-4 arm = m02 + one change.
#   hcap  (P5) --hcap 1:110,4:117,6:41,8:42,10:40 (caps from hcap_caps.py -> logs/hcap_caps.txt)
#   rec33 (P4) --rec_q 0.33 --rec_days 3 (Sep 17-19 MSK)
#   rho70 (P3) --rho_file rho/excess_r<root>.npz --rho_keep 0.7; a rho70 job is skipped (not claimed) unless
#         rho/excess_r<root>.npz AND the pre-flight marker rho/PREFLIGHT_PASS (rho_preflight.py) exist.
#   J="hcap:101 rec33:101 rho70:101 hcap:102 rec33:102 rho70:102 ..."; LANES=1 setsid nohup bash nn_x/r6/cd_lane.sh <g> "$J" ...
set -uo pipefail
R6=<repo>/nn_x/r6
PY=<repo>/.venv/bin/python
gpu=${1:?gpu (4-7)}; jobs=${2:?"job list arm:root ..."}; DRY=${DRY:-0}
case $gpu in 0|1|2|3|4|5|6|7) ;; *) echo "refusing GPU $gpu" >&2; exit 1 ;; esac
# F7 cooldown, explicit loss_norm feed (nn/run_f6cd.sh omitted it: the F6cd members trained with 'batch')
CD="--epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise \
--aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.1 --loss_norm feed --holdout 30000 --select_on holdout \
--eval_every 1000 --seed ${CDSEED:-811}"  # lead 10:55: CDSEED varies the data order of extra branches
flags() {  # appended LAST (argparse: last occurrence wins); $1 = arm, $2 = root (wave 4: rho70 reads its root's file)
  case $1 in
    c0) echo "" ;;                         # control = F7's cooldown, new data order (also an R1 seed-branch member)
    m02) echo "--cooldown_mix 0.02" ;;     # pool dose matched to the 2-epoch recipe (~0.93 extra pool passes)
    w10) echo "--wd 0.1" ;;                # 5x decoupled weight decay during the cooldown
    m0) echo "--cooldown_mix 0" ;;         # wave 2: no pool replay
    aa5) echo "--aux_anneal 0.5" ;;        # wave 2: K9 aux kept at 0.5 through the cooldown
    d0) echo "--drop_cd 0" ;;              # wave 2: every dropout site (incl. attention probs) 0 from the cooldown start
    d20) echo "--drop_cd 0.2" ;;           # wave 2: every dropout site 0.2 from the cooldown start
    pp02) echo "--pool_p 0.2" ;;           # wave 2: pool p_random > 0.2 (M>=50, holdout excluded): 197,477 vs 134,118
    aug) echo "--aug_cand 0.6 --aug_hist 0.15" ;;  # wave 2 (r6_external R2): candidate subset + history drop, pool only
    cf4) echo "--cooldown_mix 0.02 --epochs 4" ;;  # wave 3 (lead 12:46 11-12): run with SNAP=60000 -> 40% cooldown, total ~99.3K steps (matched)
    e3) echo "--cooldown_mix 0.02 --epochs 3" ;;   # wave 3: 3-epoch m02 cooldown from @77500 (total ~107K steps, +10% compute)
    s2c2) echo "--cooldown_mix 0.02 --epochs 4 --cooldown_frac 0.5" ;;  # wave 3 (lead 12:51): 2 more stable epochs at lr 2e-3, then the 2-epoch m02 cooldown (~117K steps = 12 epochs)
    hcap) echo "--cooldown_mix 0.02 --hcap 1:110,4:117,6:41,8:42,10:40" ;;  # wave 4 P5: val/test history-format caps in the cooldown (+ selection holdout)
    rec33) echo "--cooldown_mix 0.02 --rec_q 0.33 --rec_days 3" ;;          # wave 4 P4: 33% of non-pool cooldown batches from Sep 17-19
    rho70) echo "--cooldown_mix 0.02 --rho_file $R6/rho/excess_r${2:?root}.npz --rho_keep 0.7" ;;  # wave 4 P3: top-70% excess production feeds
    *) echo "unknown CD arm $1" >&2; return 1 ;;
  esac
}
LANES=${LANES:-2}
check_one_lane() {  # LANES=1: this lane adds ONE process (cap 4 per GPU) and needs ~LANE_GB free
  local n mem used total free need=${LANE_GB:-60}
  n=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  mem=$(nvidia-smi -i "$1" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  used=${mem%%,*}; total=${mem##*,}; used=${used//[^0-9]/}; total=${total//[^0-9]/}
  if [[ -z $used || -z $total ]]; then echo "refusing: cannot read memory of GPU $1 with nvidia-smi" >&2; return 1; fi
  free=$(( (total - used) / 1024 ))
  echo "GPU $1: $n compute processes, ${free} GB free (+1 lane needs ~${need} GB; cap 4 processes per GPU)"
  if (( n >= 4 )); then echo "refusing GPU $1: $n processes + 1 lane would exceed 4 per GPU" >&2; return 1; fi
  if (( free < need )); then echo "refusing GPU $1: ${free} GB free < ${need} GB" >&2; return 1; fi
}
cd "$R6" || exit 1
if [[ $DRY != 1 ]]; then
  mkdir -p logs/queue preds/holdout
  if [[ $LANES == 1 ]]; then check_one_lane "$gpu" || exit 1; else bash "$R6/queue_morning.sh" check "$gpu" || exit 1; fi
fi
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
    POLARS_MAX_THREADS=12
for j in $jobs; do
  arm=${j%%:*}; root=${j##*:}; v=r6_cd_${arm}_r${root}
  # lead 10:55 11-12: SNAP=<step> re-cools from an earlier pre-cooldown snapshot; default 77500 keeps the old names
  [[ ${SNAP:-77500} == 77500 ]] || v=r6_cd_${arm}_p$(( SNAP / 1000 ))k_r${root}
  # lead 11:50 11-12: SRC=<tag> re-cools another family's snapshots nn_x/r6/ckpt/r6_<tag>_e10_s<root>@<SNAP>.pt
  [[ -z ${SRC:-} ]] || v=r6_cd_${arm}_${SRC}_r${root}
  # lead 15:18 11-12: CDSEED != 811 (cooldown data-order seed) gets its own name so it never collides with seed-811 runs
  [[ ${CDSEED:-811} == 811 ]] || v=${v}_cs${CDSEED}
  f=$(flags "$arm" "$root") || exit 1
  if [[ $arm == rho70 && $DRY != 1 ]] && [[ ! -f $R6/rho/excess_r${root}.npz || ! -f $R6/rho/PREFLIGHT_PASS ]]; then
    echo "$(date -u +%FT%TZ) SKIP $v: rho/excess_r${root}.npz or rho/PREFLIGHT_PASS missing (P3 pre-flight gate)"; continue; fi
  init=<repo>/nn/ckpt/v17_pre2s_s${root}@${SNAP:-77500}.pt
  [[ -z ${SRC:-} ]] || init=$R6/ckpt/r6_${SRC}_e10_s${root}@${SNAP:-77500}.pt
  [[ -f $init ]] || { echo "missing snapshot $init" >&2; exit 1; }
  cmd="$PY train.py --variant $v --init $init $CD $f"
  if [[ $DRY == 1 ]]; then echo "GPU $gpu: $cmd"; echo "GPU $gpu: $PY holdout_eval.py $v"; continue; fi
  mkdir "logs/queue/cd_claim_$v" 2>/dev/null || continue   # claimed by another lane
  echo "$(date -u +%FT%TZ) START $v on GPU $gpu"
  if [[ ! -f logs/$v.json ]]; then                          # train.py writes the json last
    exec 9>"logs/queue/load_gpu${gpu}.lock"; flock 9
    $cmd > "logs/$v.stdout" 2>&1 &
    p=$!
    until grep -q "steps/epoch" "logs/$v.stdout" 2>/dev/null || ! kill -0 "$p" 2>/dev/null; do sleep 15; done
    flock -u 9; exec 9>&-
    wait "$p" || { echo "FAILED train $v"; touch "logs/queue/cd_failed_$v"; continue; }
  fi
  if [[ ! -f preds/holdout/nn_${v}_holdout.npy ]]; then
    exec 9>"logs/queue/load_gpu${gpu}.lock"; flock 9
    $PY holdout_eval.py "$v" > "logs/${v}_holdout.stdout" 2>&1 || echo "FAILED holdout_eval $v"
    flock -u 9; exec 9>&-
  fi
  echo "$(date -u +%FT%TZ) DONE $v"
done
