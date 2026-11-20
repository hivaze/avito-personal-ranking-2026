#!/usr/bin/env bash
# Runner of the 2-epoch ablation screen. One lane works through a shared job list on one GPU; several
# lanes split the list through atomic mkdir claims. Each job is the base recipe plus the flags of one arm.
# Run as: bash nn_x/abl/abl_lane.sh <gpu> "<arm>:<seed> <arm>:<seed> ..." with seeds 23 and 24.
# The first screen (2 base runs + 12 arms x 2 seeds = 26 runs) started at 00:24 UTC on 2026-11-13 on
# GPUs N and 7, three lanes each; a run took 11 to 55 minutes.
# Produced logs/abl_<arm>_s<seed>.{log,json}, checkpoints and val / holdout predictions per run.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/abl/abl_lane.sh
#
# nn_x/abl ABLATION-SCREEN lane (ledger nn_x/abl/ledger/ledger.md section 4; 2-epoch paired screen of the current recipe's
# components). Written 2026-11-13 ~00:10 UTC, modelled on nn_x/w5/lane.sh stage A.
# Second wave 2026-11-13 ~04:00 UTC (ABLATION_PLAN_1113 section 3.3, ledger section 6): arms retr_cat and xdrop_ctxhn_hs
# appended, ALLOWED default extended to "5 7 3" (this file was replaced by mv, never edited in place).
# pop1 arm 2026-11-13 ~06:10 UTC (nn_x/codes/followup/RESULT.txt section (4), ledger section 7): arm pop1 appended (replaced by
# mv again; nothing else changed).
# Runs a shared job list sequentially on ONE GPU; several lanes given the same list split it through atomic mkdir claims.
#
#   bash nn_x/abl/abl_lane.sh <gpu> "<arm>:<seed> <arm>:<seed> ..."       seed 23 | 24 -> variant abl_<arm>_s<seed>
#   DRY=1 bash nn_x/abl/abl_lane.sh <gpu> "<jobs>"      print the train.py commands only (no GPU check, no process)
#   ABL_SMOKE=1 ...                                     plumbing smoke: variant abl_smoke_*, 20K train feeds, 1 epoch,
#                                                       evals every 25 steps, select_min_step 25 (never for the screen)
# Base = nn_x/w5/lane.sh A_BASE (the nn_x/r7 BASE2 2-epoch screen recipe with --cooldown_mix 0.02) + the wave-5 hygiene
# flags HYG, held as a LIST so that REMOVAL arms can DROP store_true flags (appending cannot switch them off) while
# value arms APPEND last (argparse: the last occurrence wins). Arms (ledger section 4, rank order):
#   base         (reference, seeds 23 / 24; the stage-A base of wave 5 never ran)
#   no_aux       ABLATION  + --aux_pos 0                     (K9 production-order aux head off; --aux_anneal inert)
#   no_retr      ABLATION  - --retr_m 16 - --attn_gate       (history retrieval + gate off)
#   retr32       SCALE-UP  + --retr_m 32
#   retr32u      SCALE-UP  + --retr_m 32 --retr_uniq --retr_tcnt
#   xdrop_pop    ABLATION  + --xt_exclude <8 popularity / co-visitation columns>
#   xdrop_ctxnd  ABLATION  + --xt_exclude <10 ctx_nd columns>
#   no_sim       ABLATION  - --sim_bias
#   no_domain    ABLATION  - --domain                        (p_random still loads through --holdout)
#   xrel_more    SCALE-UP  + --xt_rel_extra seq_cv_pop,it_hist_et1_luo,ctx_nd_knn_h,ctx_nd_ucent
#   lnorm_batch  ABLATION  + --loss_norm batch
#   blocks3      SCALE-UP  + --blocks 3
#   no_set       ABLATION  + --no_set
# Second wave (ledger section 6):
#   retr_cat       SCALE-UP  + --retr_cat 16                 (second HistRetrieval over same-category history items, zero-init Z)
#   xdrop_ctxhn_hs ABLATION  + --xt_exclude <6 ctx_hn + 4 ctx_hs columns>
#   xdrop_pre2     ABLATION  + --xt_exclude <the 6 pre2* pre-trained-encoder columns incl. has:pre2_fe> (coordinator request 04:1x UTC)
# pop1 (ledger section 7):
#   pop1           SCALE-UP  + --xt_exclude <the 8 popularity columns, exactly as xdrop_pop> --xt_add xp_imp_c50
#                  (exposure input log1p(min(tte_it_imp, 50)); primary reference arm = xdrop_pop, see abl_eval.py)
# pop2 (ledger section 8; 2026-11-13 ~14:30 UTC, POPULARITY-VIEW FAMILIES follow-up; replaced by mv again, nothing else changed):
#   pop2           SCALE-UP  + --xt_exclude <the 8 popularity columns, exactly as xdrop_pop> --xt_add xp_ctr_c
#                  (click-rate input clip(log(tte_it_ctr / 0.0159), -2, 2); primary reference arm = xdrop_pop, see abl_eval.py)
# time (ledger section 9; 2026-11-13 ~19:15 UTC, TIME arm, CURRENT_STATUS Log 19:08; replaced by mv again, nothing else changed):
#   time           SCALE-UP  + --xt_add tm_weekend,tm_hsin,tm_hcos        (the feed's Moscow weekend flag and sin / cos of the
#                  hour-of-day fraction, nn_x/abl/time/build_tm.py; the base recipe with nothing excluded; reference = base;
#                  primary read = the holdout WEEKEND slice, nn_x/abl/time/abl_eval_time.py)
# pre2noid (ledger section 10.A; 2026-11-14 ~00:10 UTC, ENCODER-DOSE (A) in-ranker dose; replaced by mv again, nothing else changed):
#   pre2noid       SWAP      + --xt_exclude <the 6 pre2* columns, exactly as xdrop_pre2> --xt_add <the 6 pre2n_* ID-free columns>
#                            --xt_add_relonly pre2n_fe_s,pre2n_new_s,pre2n_hmax32   (same layout as the base: the ID-free encoder
#                            scores take the live pre2 columns' indices 28-33, rank / z-only treatment of the 3 raw-scale ones;
#                            read against base AND xdrop_pre2 by nn_x/abl/encdose/abl_eval_encdose.py, KILL arithmetic of ledger 10.A)
# kv64 (ledger section 10.C; ENCODER-DOSE (C) frozen hidden-state K/V probe):
#   kv64           SCALE-UP  + --kv64 1   (the frozen pre2 encoder's 64-d PCA states of every history event as a zero-init gated K/V
#                            adapter in the 2 cross blocks, encdose/kv_probe; bit-identical to the base at init; reference = base)
# GPU rules (ablation task): only GPUs in ABL_GPUS (default "5 7 3") are accepted; before EACH launch the GPU must have
#   <= 3 compute processes (< 4) and >= LANE_GB (60) GB free, checked with nvidia-smi under a per-GPU flock that is held
#   until the job has loaded its data (train.py logs "steps/epoch"). A GPU that does not qualify is polled every 120 s
#   (WAIT_MAX_H hours, default 6, then the lane stops). CUDA_VISIBLE_DEVICES = exactly that id; CPU threads 12.
#   PIDs (the train.py process, verified through /proc/<pid>/cmdline and pgrep -f with the exact variant) go to
#   nn_x/abl/RUNNING.txt; stop jobs by exact PID only. Finished jobs (logs/<v>.json + selected and @last holdout preds)
#   are never re-run; a failed job leaves logs/queue/failed_<v>.
set -uo pipefail
ABL=<repo>/nn_x/abl
PY=<repo>/.venv/bin/python
gpu=${1:?gpu}; jobs=${2:?"job list <arm>:<seed> ..."}; DRY=${DRY:-0}
ALLOWED=${ABL_GPUS:-"5 7 3"}
[[ " $ALLOWED " == *" $gpu "* ]] || { echo "abl_lane.sh: refusing GPU $gpu (allowed: $ALLOWED)" >&2; exit 1; }
A_BASE=(--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000
  --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2
  --cooldown_mix 0.02 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 1000 --no_test)
HYG=(--save_last --save_holdout --select_min_step 12000 --cliplog 50)
POP=seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo
CTXND=ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_h,ctx_nd_knn_h_dev,ctx_nd_knn_u,ctx_nd_knn_u_dev,ctx_nd_max,ctx_nd_top5h,ctx_nd_top5u,ctx_nd_ucent
CTXHNHS=ctx_hn_top1w,ctx_hn_rrw,ctx_hn_bestrk,ctx_hn_margin,ctx_hn_soft05,ctx_hn_e1_rrw,ctx_hs_mutual,ctx_hs_k128,ctx_hs_e1,ctx_hs_t10
XREL=seq_cv_pop,it_hist_et1_luo,ctx_nd_knn_h,ctx_nd_ucent
PRE2=pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32,has:pre2_fe
PRE2N=pre2n_fe_z,pre2n_new_z,pre2n_fe_s,pre2n_new_s,pre2n_hmax32,has:pre2n_fe
PRE2N_RO=pre2n_fe_s,pre2n_new_s,pre2n_hmax32
ARMS="base no_aux no_retr retr32 retr32u xdrop_pop xdrop_ctxnd no_sim no_domain xrel_more lnorm_batch blocks3 no_set retr_cat xdrop_ctxhn_hs xdrop_pre2 pop1 pop2 time pre2noid kv64"
arm_spec() {  # $1 = arm -> DROP (flags removed from A_BASE together with their values) and ADD (appended LAST)
  DROP=""; ADD=""
  case $1 in
    base) ;;
    no_aux) ADD="--aux_pos 0" ;;
    no_retr) DROP="--retr_m --attn_gate" ;;
    retr32) ADD="--retr_m 32" ;;
    retr32u) ADD="--retr_m 32 --retr_uniq --retr_tcnt" ;;
    xdrop_pop) ADD="--xt_exclude $POP" ;;
    xdrop_ctxnd) ADD="--xt_exclude $CTXND" ;;
    no_sim) DROP="--sim_bias" ;;
    no_domain) DROP="--domain" ;;
    xrel_more) ADD="--xt_rel_extra $XREL" ;;
    lnorm_batch) ADD="--loss_norm batch" ;;
    blocks3) ADD="--blocks 3" ;;
    no_set) ADD="--no_set" ;;
    retr_cat) ADD="--retr_cat 16" ;;
    xdrop_ctxhn_hs) ADD="--xt_exclude $CTXHNHS" ;;
    xdrop_pre2) ADD="--xt_exclude $PRE2" ;;
    pop1) ADD="--xt_exclude $POP --xt_add xp_imp_c50" ;;
    pop2) ADD="--xt_exclude $POP --xt_add xp_ctr_c" ;;
    time) ADD="--xt_add tm_weekend,tm_hsin,tm_hcos" ;;
    pre2noid) ADD="--xt_exclude $PRE2 --xt_add $PRE2N --xt_add_relonly $PRE2N_RO" ;;
    kv64) ADD="--kv64 1" ;;
    *) echo "abl_lane.sh: unknown arm $1 (arms: $ARMS)" >&2; return 1 ;;
  esac
}
build_args() {  # $1 = seed: A_BASE minus the DROP flags (each with its following non-flag values), + HYG + seed + ADD
  local out=() skip=0 t
  for t in "${A_BASE[@]}"; do
    if [[ $t == --* ]]; then
      skip=0
      [[ -n $DROP && " $DROP " == *" $t "* ]] && skip=1
    fi
    (( skip )) && continue
    out+=("$t")
  done
  echo "${out[*]} ${HYG[*]} --seed $1 $ADD"
}
check_gpu() {  # $1 = gpu; 0 iff <= 3 compute processes and >= LANE_GB GB free
  local n mem used total free need=${LANE_GB:-60}
  n=$(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  mem=$(nvidia-smi -i "$1" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  used=${mem%%,*}; total=${mem##*,}; used=${used//[^0-9]/}; total=${total//[^0-9]/}
  if [[ -z $used || -z $total ]]; then echo "$(date -u +%FT%TZ) GPU $1: cannot read nvidia-smi"; return 1; fi
  free=$(( (total - used) / 1024 ))
  echo "$(date -u +%FT%TZ) GPU $1: $n compute processes, ${free} GB free (need <= 3 and >= ${need} GB)"
  (( n <= 3 && free >= need ))
}
cd "$ABL" || exit 1
mkdir -p logs/queue logs/locks preds/holdout ckpt
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
    POLARS_MAX_THREADS=12
RUN=$ABL/RUNNING.txt
for j in $jobs; do
  IFS=: read -r arm seed x <<< "$j"
  [[ -z $x ]] || { echo "abl_lane.sh: bad job $j (want <arm>:<seed>)" >&2; exit 1; }
  [[ $seed == 23 || $seed == 24 ]] || { echo "abl_lane.sh: screen seeds are 23 / 24 (got $seed in $j)" >&2; exit 1; }
  arm_spec "$arm" || exit 1
  v=abl_${arm}_s${seed}; args=$(build_args "$seed")
  if [[ ${ABL_SMOKE:-0} == 1 ]]; then
    v=${v/abl_/abl_smoke_}; args="$args --train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25 --cliplog 10"
  fi
  cmd="$PY train.py --variant $v $args"
  if [[ $DRY == 1 ]]; then echo "GPU $gpu: $cmd"; continue; fi
  done_() { [[ -f logs/$v.json && -f preds/holdout/nn_${v}_holdout.npy && -f preds/holdout/nn_${v}@last_holdout.npy ]]; }
  if done_; then echo "$(date -u +%FT%TZ) $v already done"; continue; fi
  mkdir "logs/queue/claim_$v" 2>/dev/null || continue   # claimed by another lane
  exec 9>"logs/locks/gpu${gpu}.lock"; flock 9
  t0=$(date +%s)
  until check_gpu "$gpu"; do
    (( $(date +%s) - t0 > ${WAIT_MAX_H:-6} * 3600 )) && { echo "$(date -u +%FT%TZ) GPU $gpu never qualified: lane stops"; \
      rmdir "logs/queue/claim_$v"; flock -u 9; exit 3; }
    sleep 120
  done
  echo "$(date -u +%FT%TZ) START $v on GPU $gpu: $cmd"
  $cmd > "logs/$v.stdout" 2>&1 &
  p=$!
  sleep 2
  # exact-PID check: $p must be the train.py process of THIS variant (its /proc cmdline), and pgrep must agree
  pc=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
  pg=$(pgrep -f -- "train.py --variant $v " | tr '\n' ' ')
  [[ $pc == *"train.py --variant $v "* ]] && ok=verified || ok=UNVERIFIED
  echo "$(date -u +%FT%TZ) START pid=$p ($ok; pgrep: ${pg:-none}) lane_pid=$$ variant=$v gpu=$gpu job=$j" >> "$RUN"
  until grep -q "steps/epoch" "logs/$v.stdout" 2>/dev/null || ! kill -0 "$p" 2>/dev/null; do sleep 15; done
  flock -u 9; exec 9>&-
  rc=0; wait "$p" || rc=$?
  echo "$(date -u +%FT%TZ) END pid=$p variant=$v rc=$rc" >> "$RUN"
  if (( rc != 0 )); then echo "$(date -u +%FT%TZ) FAILED $v (rc $rc)"; touch "logs/queue/failed_$v"; continue; fi
  if [[ ! -f preds/holdout/nn_${v}_holdout.npy ]]; then  # fallback (save_holdout writes it): holdout_eval.py, same lock
    exec 9>"logs/locks/gpu${gpu}.lock"; flock 9
    $PY holdout_eval.py "$v" > "logs/${v}_holdout.stdout" 2>&1 || echo "FAILED holdout_eval $v"
    flock -u 9; exec 9>&-
  fi
  echo "$(date -u +%FT%TZ) DONE $v"
done
