#!/usr/bin/env bash
# The runner of the waves. One lane takes a list of jobs and runs them one after another on one
# GPU; several lanes given the same list share it through atomic mkdir claims. A job is either a
# 2-epoch run from scratch (stage A), a 2-epoch cooldown-only run from a stored checkpoint (stage
# B, the 're-cool'), or a 10-epoch run (stage E). Every arm is the base command plus one flag; the
# comment block at the top lists the arms of waves 5 to 8 with the time each was added.
# How it was run: bash nn_x/w5/lane.sh <gpu> "<job> <job> ...", started by launch_w5.sh in the
# same lab folder. Example of a logged job (lab/nn_x/w5/logs/lane_w5_gpu0_1.out): train.py --variant
# w5_cd_m02ref_r101 --init .../v17_pre2s_s101@77500.pt --epochs 2 --lr 2e-3 ... --seed 1001.
# What it produced: one checkpoint, log and prediction set per job under lab/nn_x/w5/.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w5/lane.sh
# nn_x/w5 WAVE-5 lane (reports/INSIGHTS_1112.md 3.1 items 1 TRUNC-PROD, 3 AUXF, 8 hygiene). Written 2026-11-12 ~19:10 UTC.
# Runs a shared job list sequentially on ONE GPU; several lanes given the same list split it through atomic mkdir claims.
#
#   bash nn_x/w5/lane.sh <gpu> "<job> <job> ..."        job = A:<arm>:<seed> | B:<arm>:<root>
#   DRY=1 bash nn_x/w5/lane.sh <gpu> "<jobs>"           print the train.py commands only (no GPU check, no process)
#   W5_SMOKE=1 ...                                      plumbing smoke: variant w5_smoke_*, 20K train feeds, 1 epoch,
#                                                       evals every 25 steps, select_min_step 25 (never for the stages)
# Stage A (2-epoch from-scratch screen = nn_x/r7 BASE2 with --cooldown_mix 0.02 instead of 0.1), arms base | trunc | auxf,
#   seeds 23 / 24 -> variant w5_<arm>_s<seed>.
# Stage B (CD harness = nn_x/r6/cd_lane.sh CD + m02, re-cool of nn/ckpt/v17_pre2s_s<root>@77500.pt, read-only), arms
#   m02ref | truncp | auxf, roots 101-107 -> variant w5_cd_<arm>_r<root>, seed CDSEED = 900 + root for EVERY arm (item 8).
# Every job adds the wave-5 hygiene flags: --save_last (last checkpoint + val / holdout (/ test) preds), --save_holdout,
#   --select_min_step 12000 (judge at k >= 12k), --cliplog 50 (pre-clip grad norm, main / aux grad-norm split).
# Arm flags: trunc / truncp = --trunc_off 1.0; auxf = --aux_norm feed (C 1.695); base / m02ref = none.
#   (added 11-12 ~20:30 UTC) stage E trunc2 / stage B truncp2 = --trunc_off 1.0 --trunc_lifts $TL: TRUNC-PROD with the
#   per-M-bucket production lifts ([1,16] [17,32] [33,49] [50,inf); review/make_trunc_lifts.py) instead of M<50 / M>=50.
#   (added 11-12 ~22:10 UTC) stage B hzcd / stage E hzcd = --hzcd (cd_arms6.py, INSIGHTS item 6): cooldown training batches
#   read it_hist_hz_* (data/feats_new/train) in place of the it_hist_*_luo inputs; B:hzcd = m02ref + --hzcd, E:hzcd = base + --hzcd.
#   (added 11-12 ~23:35 UTC) stage B valcd = --val_mix 0.01 --val_fold <fold> (cd_arms7.py, idea D1 val-in-cooldown): job
#   B:valcd:<root>:<fold> (fold 0-4) -> variant w5_cd_valcd_r<root>_f<fold> = m02ref + the labelled val feeds of the other 4
#   user-grouped folds as an additive cooldown pool (p 0.01 of the non-pool steps); CDSEED 900 + root; read-out w5_eval_valcd.py.
#   (added 11-13 ~03:55 UTC, wave 6 = ABLATION_PLAN_1113 3.3 rows 'no_aux load-bearing' and 'always m01 / m04') stage B aa0 = --aux_anneal 0
#   (K9 aux weight annealed 0.5 -> 0 over the re-cool instead of 0.5 -> 0.1; aux head stays active), ap1 = --aux_pos 1.0 (listwise aux at
#   double weight), m01 = --cooldown_mix 0.01, m04 = --cooldown_mix 0.04; all = m02ref + ONE flag, roots 101-104, CDSEED 900 + root,
#   paired vs the existing w5_cd_m02ref_r<root>; rule logs/B_wave6_predeclared.txt, read-out w5_eval_wave6.py -> logs/w5_eval_B_wave6.txt.
#   (added 11-13 ~05:30 UTC, wave 7 = cooldown regularisers, cd_arms8.py, KL probe nn_x/tricks/rdrop_cd_kl) stage B rdrop = --rdrop_cd 1.0
#   (second dropout pass + symmetric KL), sam = --sam_cd 0.05 (one-step SAM, g2 at w + rho g1/||g1||); both = m02ref + ONE flag, roots
#   101-104, CDSEED 900 + root, vs the existing w5_cd_m02ref_r<root>; rule logs/B_wave7_predeclared.txt, read-out w5_eval_wave7.py.
#   (added 11-13 ~07:35 UTC, AP1 CONFIRMATION, CURRENT_STATUS 07:28) stage E ap1 = --aux_pos 1.0 (E:ap1:951 / 952 = E:base + double-weight
#   listwise K9 aux, paired vs w5_e10_base_s951 / 952; rule logs/E_ap1_predeclared.txt, read-out w5_eval_e_ap1.py); B:ap1 roots 105-107 + their
#   B:m02ref references complete the 7-root family (information read-out appended to logs/w5_eval_B_wave6.txt).
#   (added 11-13 ~09:25 UTC, E-EPOCHS dose-response, logs/E_epochs_predeclared.txt) stage E e15 / e20 / e30 = --epochs 15 / 20 / 30 (E_BASE + HYG,
#   the epochs flag appended last so it overrides --epochs 10; WSD cooldown_frac 0.2 scales with the run); e20 seeds 951-954 = decision arm
#   (E rule + chain_e.sh), e15 / e30 seeds 951/952 = information; read-outs scratchpad e_readout.py e20 / w5_eval_e_ap1.py e15|e30 951,952.
#   (added 11-13 ~10:55 UTC, E-BATCH, logs/E_batch_predeclared.txt) stage E bs4 = 4x batch at 10 epochs (--cand_budget 131072 --max_feeds 2048,
#   eval_every 625, select_min_step 3000, save_steps 17500,19375), bs4lr4 = bs4 + --lr 4e-3; seeds 951/952 vs w5_e10_base; read-out w5_eval_e_ap1.py <arm> 951,952.
#   (added 11-13 ~11:45 UTC, WAVE 8 = cooldown batch size, logs/B_wave8_predeclared.txt) stage B cbs05 = half batch in the re-cool (--cand_budget 16384
#   --max_feeds 256 --eval_every 2000 --select_min_step 24000, ~2x cooldown steps), cbs2 = double batch (--cand_budget 65536 --max_feeds 1024 --eval_every 500
#   --select_min_step 6000, ~2x fewer steps); roots 101-104 vs the existing w5_cd_m02ref_r<root>; read-out w5_eval_wave8.py.
#
# GPU rules (wave-5 task): only GPUs in W5_GPUS (default "<ids>") are accepted; before EACH launch the GPU must have
#   <= 3 compute processes (<= 4 with this one) and >= LANE_GB (60) GB free, checked with nvidia-smi under a per-GPU flock
#   that is held until the job has loaded its data (train.py logs "steps/epoch"). A GPU that does not qualify is polled
#   every 120 s (WAIT_MAX_H hours, default 6, then the lane stops). CUDA_VISIBLE_DEVICES = exactly that id; CPU threads 12.
#   PIDs go to logs/RUNNING_w5.txt; stop jobs by exact PID only. Finished jobs (logs/<v>.json + selected and @last holdout
#   preds) are never re-run; a failed job leaves logs/queue/failed_<v>.
set -uo pipefail
W5=<repo>/nn_x/w5
PY=<repo>/.venv/bin/python
gpu=${1:?gpu}; jobs=${2:?"job list A:<arm>:<seed> | B:<arm>:<root> ..."}; DRY=${DRY:-0}
ALLOWED=${W5_GPUS:-"0 1 2 4"}
[[ " $ALLOWED " == *" $gpu "* ]] || { echo "lane.sh: refusing GPU $gpu (allowed: $ALLOWED)" >&2; exit 1; }
A_BASE="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
--select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 \
--cooldown_mix 0.02 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 1000 --no_test"
B_BASE="--epochs 2 --lr 2e-3 --warmup 0 --sched wsd --cooldown_frac 1.0 --muon --aux_pos 0.5 --aux_mode listwise \
--aux_tau 10 --aux_anneal 0.1 --cooldown_mix 0.02 --loss_norm feed --holdout 30000 --select_on holdout --eval_every 1000"
#   (added 11-14 ~20:25 UTC, RECHECK_1114 follow-ups, rules in logs/E_kd_predeclared.txt 20:25Z block) u20a/b/c (unlabelled-fold KD, needs --xfold_unlab patch),
#   kwh30/kwh10 (history-drop function matching), kwaux0/kwaux1, kwwd05/kwwd10, kwm05, kwcf35, kd08w256, kw256d2, kwr32, kwh6, kwsb2 (never varied under KD).
HYG="--save_last --save_holdout --select_min_step 12000 --cliplog 50"
# lead 19:4x 11-12 (after the insights review: TRUNC's gain builds over training -> 10-epoch from-scratch PAIRS):
# stage E = the F7 m02 recipe (nn/run_final15.sh kind f7 flags) at 10 epochs, seeds 951-954, arms base | trunc | auxf
E_BASE="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
--select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd \
--cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500"
TL=<repo>/nn_x/w5/trunc_lifts_mbucket.json
arm_flags() {  # $1 = stage, $2 = arm; appended LAST (argparse: the last occurrence wins)
  case $1:$2 in
    A:base|B:m02ref|E:base) echo "" ;;
    A:trunc|B:truncp|E:trunc) echo "--trunc_off 1.0" ;;
    E:trunc2|B:truncp2) [[ -f $TL ]] || { echo "lane.sh: missing $TL" >&2; return 1; }
      echo "--trunc_off 1.0 --trunc_lifts $TL" ;;
    A:auxf|B:auxf|E:auxf) echo "--aux_norm feed" ;;
    B:hzcd|E:hzcd) echo "--hzcd" ;;
    B:valcd) echo "--val_mix 0.01" ;;
    B:aa0) echo "--aux_anneal 0" ;;
    B:ap1|E:ap1) echo "--aux_pos 1.0" ;;
    B:m01) echo "--cooldown_mix 0.01" ;;
    B:m04) echo "--cooldown_mix 0.04" ;;
    B:rdrop) echo "--rdrop_cd 1.0" ;;
    B:sam) echo "--sam_cd 0.05" ;;
    B:cbs05) echo "--cand_budget 16384 --max_feeds 256 --eval_every 2000 --select_min_step 24000" ;;
    B:cbs2) echo "--cand_budget 65536 --max_feeds 1024 --eval_every 500 --select_min_step 6000" ;;
    B:cbs025) echo "--cand_budget 8192 --max_feeds 128 --eval_every 4000 --select_min_step 48000" ;;  # wave-8 addendum 1 (information arm)
    B:xp2) echo "--prand2 <repo>/eda/labels_position/out/train_prandom_v2.npz" ;;
    B:xp2m05) echo "--prand2 <repo>/eda/labels_position/out/train_prandom_v2.npz --cooldown_mix 0.05" ;;
    B:xp2m10) echo "--prand2 <repo>/eda/labels_position/out/train_prandom_v2.npz --cooldown_mix 0.10" ;;
    E:xp2m05) echo "--prand2 <repo>/eda/labels_position/out/train_prandom_v2.npz --cooldown_mix 0.05" ;;
    E:xp2) echo "--prand2 <repo>/eda/labels_position/out/train_prandom_v2.npz" ;;
    E:kd05) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5" ;;
    E:kd08) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.8" ;;
    E:kd10) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 1.0" ;;
    E:kd08t25) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean25.npy --kd_w 0.8" ;;
    E:kd08w192) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.8 --d 192" ;;
    E:kd03) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.3" ;;
    E:kd05w192) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192" ;;
    E:kd05t25) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean25.npy --kd_w 0.5" ;;
    E:kd05t25w192) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean25.npy --kd_w 0.5 --d 192" ;;
    E:kd05t25w256) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean25.npy --kd_w 0.5 --d 256" ;;
    E:kd05g2w192) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_w8g1.npy --kd_w 0.5 --d 192" ;;
    E:kd03g2w192) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_w8g1.npy --kd_w 0.3 --d 192" ;;
    E:kw_drop0) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --drop 0.0" ;;
    E:kw_e20) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --epochs 20" ;;
    E:kw_id) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --id_buckets 500000 --id_drop 0.5" ;;
    E:kw_hl2) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --hist_layers 2" ;;
    E:kw_ff4) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --ff 4" ;;
    E:xf0) echo "--xfold 0" ;;
    E:xf1) echo "--xfold 1" ;;
    E:xf2) echo "--xfold 2" ;;
    E:xf3) echo "--xfold 3" ;;
    E:xf4) echo "--xfold 4" ;;
    E:kx05) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_oof.npy --d 192 --kd_w 0.5" ;;
    E:kx08) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_oof.npy --d 192 --kd_w 0.8" ;;
    E:kd05w192gru) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --arch gru" ;;
    E:kdmixw192) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mix25w8.npy --kd_w 0.5 --d 192" ;;
    E:kd05w224) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 224" ;;
    E:kwsub5) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --aug_all --aug_cand 0.5" ;;
    E:kwsub2) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --aug_all --aug_cand 0.2" ;;
    E:kwT07) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --kd_T 0.7" ;;
    E:kwT15) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --kd_T 1.5" ;;
    E:u20a) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_xf0.npy --kd_T 3 --kd_w 0.5 --d 192 --xfold 0" ;;
    E:u20b) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_xf0.npy --kd_T 3 --kd_w 0.5 --d 192 --xfold 0 --xfold_unlab" ;;
    E:u20c) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_xf0.npy --kd_T 3 --kd_w 0.5 --d 192" ;;
    E:kwh30) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --aug_all --aug_hist 0.3" ;;
    E:kwh10) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --aug_all --aug_hist 0.1" ;;
    E:kwaux0) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --aux_pos 0" ;;
    E:kwaux1) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --aux_pos 1.0" ;;
    E:kwwd05) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --wd 0.05" ;;
    E:kwwd10) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --wd 0.1" ;;
    E:kwm05) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --cooldown_mix 0.05" ;;
    E:kwcf35) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --cooldown_frac 0.35" ;;
    E:kd08w256) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.8 --d 256" ;;
    E:kw256d2) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 256 --drop 0.2" ;;
    E:kwr32) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --retr_m 32" ;;
    E:kwh6) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --heads 6" ;;
    E:kwsb2) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --sim_bias 2" ;;
    E:kd05t25w192b3) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean25.npy --kd_w 0.5 --d 192 --blocks 3" ;;
    E:e15) echo "--epochs 15" ;;
    E:kwxd15) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --xt_drop 0.15" ;;   # (11-15 03:40 UTC) popularity-input dropout under KD (veto audit item 4)
    E:kwxd30) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --xt_drop 0.30" ;;
    E:kwd2) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 --d 192 --drop 0.2" ;;   # SYNTHESIS 3.1 item 4
    # (added 11-14 ~21:50 UTC) stages K / L = KD RE-COOL HARNESS at d192 (RECHECK_1114 skeptic_feasibility_value 7.2.1; rules in
    #   logs/E_kd_predeclared.txt "RE-COOL HARNESS" block): cooldown phase only (19,670 steps = B_BASE) from the stored plateau snapshot
    #   w5_e10_kd05w192_s<seed>@77500 (K, teacher_mean11) or w5_e10_kd05t25w192_s<seed>@77500 (L, teacher_mean25), seeds 951-958.
    K:ref|L:ref) echo "" ;;
    K:hz|L:hz) echo "--hzcd" ;;   # (added 11-15 ~02:20 UTC) RE-OPENED popularity-parity cooldown under KD at d192; rule "ARM hz" in logs/E_kd_predeclared.txt
    K:th|L:th) echo "--kd_file <repo>/nn_x/w12/kd/out/teacher_half_p1.npy" ;;   # (11-15 03:40 UTC) popularity-light teacher dose (veto audit item 7); overrides the stage teacher (last flag wins)
    K:xd30|L:xd30) echo "--xt_drop_init 0.30" ;;   # (11-15 04:55 UTC) popularity-input dropout in the re-cool (staleness family; RULE hz)
    K:xd15|L:xd15) echo "--xt_drop_init 0.15" ;;
    K:xd50|L:xd50) echo "--xt_drop_init 0.50" ;;
    K:hzxd|L:hzxd) echo "--hzcd --xt_drop_init 0.30" ;;
    K:s3|L:s3) echo "--hzcd --xt_drop_init 0.30 --kd_file <repo>/nn_x/w12/kd/out/teacher_half_p1.npy" ;;   # speculative bundle hz + xd30 + th
    K:hz2|L:hz2|K:hz2L|L:hz2L) echo "--hzcd" ;;   # (11-15 05:05 UTC) hz2 = full-dose causal test-like view of the 8 popularity inputs (reports/STALE_1115/hz2.md); view dir by env, set per job below
    K:h2t|L:h2t) echo "--hzcd --kd_file <repo>/nn_x/w12/kd/out/teacher_half_p1.npy" ;;   # hz2 view + popularity-light teacher
    K:h2x|L:h2x) echo "--hzcd --xt_drop_init 0.15" ;;   # hz2 view + light popularity-input dropout
    K:hzt|L:hzt) echo "--hzcd --kd_file <repo>/nn_x/w12/kd/out/teacher_half_p1.npy" ;;   # (11-15 08:58 UTC) hz (half-dose view, HZCD_DIR unset) + popularity-light teacher; rule in logs/E_kd_predeclared.txt
    K:hzn|L:hzn|K:hznF|L:hznF|K:hznFS|L:hznFS|K:hz2n|L:hz2n|K:hz2nF|L:hz2nF|K:hz2Ln|L:hz2Ln|K:hz2LnF|L:hz2LnF) echo "--hzcd" ;;   # (11-15 07:41 UTC) hz2n family (reports/STALE_1115/hz2n.md, nn_x/w12/stale/hz2n/READY.txt): base view + NO-ID encoder score on rows made cold by the view (cd_arms6c); view dir by env below
    K:ref2|L:ref2) echo "" ;;   # second cooldown seed (2000 + seed, set in the stage branch): the harness's own A/A null (SYNTHESIS 3.1 item 1)
    M:ps) echo "" ;;            # stage M (added 11-14 ~22:25 UTC, SYNTHESIS 3.1 item 7): TRUE PHASE SPLIT = kd_w 0.8 plateau snapshot, cooldown at kd_w 0.5
    M:kw08) echo "--kd_w 0.8" ;;  # stage M control: the kd08w192 plateau re-cooled at its own kd_w 0.8
    K:kw08|L:kw08) echo "--kd_w 0.8" ;;
    K:kw03|L:kw03) echo "--kd_w 0.3" ;;
    K:kw0|L:kw0) echo "--kd_w 0" ;;
    K:m0|L:m0) echo "--cooldown_mix 0" ;;
    K:m05|L:m05) echo "--cooldown_mix 0.05" ;;
    K:m10|L:m10) echo "--cooldown_mix 0.10" ;;
    K:e4|L:e4) echo "--epochs 4" ;;
    K:e1|L:e1) echo "--epochs 1 --select_min_step 6000" ;;
    K:wd05|L:wd05) echo "--wd 0.05" ;;
    K:wd0|L:wd0) echo "--wd 0" ;;
    K:aux0|L:aux0) echo "--aux_pos 0" ;;
    K:dcd0|L:dcd0) echo "--drop_cd 0" ;;
    K:dcd2|L:dcd2) echo "--drop_cd 0.2" ;;
    K:lr1|L:lr1) echo "--lr 1e-3" ;;
    K:lr4|L:lr4) echo "--lr 4e-3" ;;
    E:e20) echo "--epochs 20" ;;
    E:e30) echo "--epochs 30" ;;
    E:bs4) echo "--cand_budget 131072 --max_feeds 2048 --eval_every 625 --select_min_step 3000 --save_steps 17500,19375" ;;
    E:bs4lr4) echo "--cand_budget 131072 --max_feeds 2048 --eval_every 625 --select_min_step 3000 --save_steps 17500,19375 --lr 4e-3" ;;
    *) echo "lane.sh: unknown arm $2 for stage $1" >&2; return 1 ;;
  esac
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
cd "$W5" || exit 1
mkdir -p logs/queue logs/locks preds/holdout ckpt
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 \
    POLARS_MAX_THREADS=12
RUN=logs/RUNNING_w5.txt
for j in $jobs; do
  IFS=: read -r st arm x y <<< "$j"
  [[ -z $y || $arm == valcd ]] || { echo "lane.sh: unexpected 4th field in job $j (only B:valcd:<root>:<fold> has one)" >&2; exit 1; }
  f=$(arm_flags "$st" "$arm") || exit 1
  if [[ $st == E ]]; then
    (( x >= 951 && x <= 962 )) || { echo "lane.sh: stage E seeds are 951-962 (got $x)" >&2; exit 1; }
    v=w5_e10_${arm}_s${x}; args="$E_BASE $HYG --seed $x $f"
  elif [[ $st == K || $st == L ]]; then  # KD re-cool harness at d192 (architecture comes from the snapshot cfg)
    (( x >= 951 && x <= 958 )) || { echo "lane.sh: stage K / L seeds are 951-958 (got $x)" >&2; exit 1; }
    if [[ $st == K ]]; then fam=kd05w192; tch=teacher_mean11; stl=k; else fam=kd05t25w192; tch=teacher_mean25; stl=l; fi
    init=$W5/ckpt/w5_e10_${fam}_s${x}@77500.pt
    [[ -f $init ]] || { echo "lane.sh: missing snapshot $init" >&2; exit 1; }
    v=w5_${stl}cd_${arm}_s${x}; cds=$((1000 + x)); [[ $arm == ref2 ]] && cds=$((2000 + x))
    args="--init $init $B_BASE $HYG --seed $cds --kd_file <repo>/nn_x/w12/kd/out/${tch}.npy --kd_w 0.5 $f"
  elif [[ $st == M ]]; then  # true phase split: kd08w192 plateau snapshot, cooldown at kd_w 0.5 (arm ps) or 0.8 (arm kw08)
    (( x >= 951 && x <= 958 )) || { echo "lane.sh: stage M seeds are 951-958 (got $x)" >&2; exit 1; }
    init=$W5/ckpt/w5_e10_kd08w192_s${x}@77500.pt
    [[ -f $init ]] || { echo "lane.sh: missing snapshot $init" >&2; exit 1; }
    v=w5_mcd_${arm}_s${x}
    args="--init $init $B_BASE $HYG --seed $((1000 + x)) --kd_file <repo>/nn_x/w12/kd/out/teacher_mean11.npy --kd_w 0.5 $f"
  elif [[ $st == A ]]; then
    [[ $x == 23 || $x == 24 ]] || { echo "lane.sh: stage A seeds are 23 / 24 (got $x)" >&2; exit 1; }
    v=w5_${arm}_s${x}; args="$A_BASE $HYG --seed $x $f"
  else
    (( x >= 101 && x <= 107 )) || { echo "lane.sh: stage B roots are 101-107 (got $x)" >&2; exit 1; }
    init=<repo>/nn/ckpt/v17_pre2s_s${x}@77500.pt
    [[ -f $init ]] || { echo "lane.sh: missing snapshot $init" >&2; exit 1; }
    v=w5_cd_${arm}_r${x}; args="--init $init $B_BASE $HYG --seed $((900 + x)) $f"
    if [[ $arm == valcd ]]; then  # B:valcd:<root>:<fold>: the held-out (OOF) val fold is part of the job / variant name
      [[ $y =~ ^[0-4]$ ]] || { echo "lane.sh: B:valcd needs a fold 0-4 (B:valcd:<root>:<fold>, got '$y')" >&2; exit 1; }
      v=${v}_f${y}; args="$args --val_fold $y"
    fi
  fi
  if [[ ${W5_SMOKE:-0} == 1 ]]; then
    v=${v/w5_/w5_smoke_}; args="$args --train_feeds 20000 --epochs 1 --eval_every 25 --select_min_step 25 --cliplog 10"
  fi
  unset HZCD_DIR HZ2_KEEP   # hz2: per job, never inherited by the next job of the lane (unset = cd_arms6 behaviour, the existing half-dose files)
  case $arm in
    hz2|h2t|h2x) export HZCD_DIR=<repo>/nn_x/w12/stale/hz2/feats   HZ2_KEEP=<repo>/nn_x/w12/stale/hz2/feats/train/hz2_keep_rows.npy ;;
    hz2L)        export HZCD_DIR=<repo>/nn_x/w12/stale/hz2/feats_L HZ2_KEEP=<repo>/nn_x/w12/stale/hz2/feats_L/train/hz2_keep_rows.npy ;;
    hzn)         export HZCD_DIR=<repo>/nn_x/w12/stale/hz2n/feats_hn ;;      # half-dose view (arm hz) + fe head without item ID on rows made cold
    hznF)        export HZCD_DIR=<repo>/nn_x/w12/stale/hz2n/feats_hnF ;;     # + new head too
    hznFS)       export HZCD_DIR=<repo>/nn_x/w12/stale/hz2n/feats_hnFS ;;    # + seq_cv_* = 0 on view-cold rows (no cold-row combination is train-only)
    hz2n)        export HZCD_DIR=<repo>/nn_x/w12/stale/hz2n/feats_n   HZ2_KEEP=<repo>/nn_x/w12/stale/hz2/feats/train/hz2_keep_rows.npy ;;
    hz2nF)       export HZCD_DIR=<repo>/nn_x/w12/stale/hz2n/feats_nF  HZ2_KEEP=<repo>/nn_x/w12/stale/hz2/feats/train/hz2_keep_rows.npy ;;
    hz2Ln)       export HZCD_DIR=<repo>/nn_x/w12/stale/hz2n/feats_Ln  HZ2_KEEP=<repo>/nn_x/w12/stale/hz2/feats_L/train/hz2_keep_rows.npy ;;
    hz2LnF)      export HZCD_DIR=<repo>/nn_x/w12/stale/hz2n/feats_LnF HZ2_KEEP=<repo>/nn_x/w12/stale/hz2/feats_L/train/hz2_keep_rows.npy ;;
  esac
  cmd="$PY train.py --variant $v $args"
  if [[ $DRY == 1 ]]; then echo "GPU $gpu: ${HZCD_DIR:+HZCD_DIR=$HZCD_DIR }$cmd"; continue; fi
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
  echo "$(date -u +%FT%TZ) START pid=$p lane_pid=$$ variant=$v gpu=$gpu job=$j" >> "$RUN"
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
