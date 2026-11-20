#!/usr/bin/env bash
# =====================================================================================================================
# PRE-DECLARED MATCHED-DATA TEST: soft-token Qwen3-0.6B ranker (layout L1) vs the nn/ ranker on IDENTICAL train subsets.
# Status: built and CPU-smoke-tested on 2026-11-12; pre-launch amendments S1/E1/I3/V0 implemented and CPU-retested the
# same day (reports/llm_build.md "Amendments"). NOT EXECUTED. Launch only on a GPU the lead has allocated.
# the GPUs allowed (user authorised some GPUs on 2026-11-12 03:35 UTC). At most 4 training processes per GPU. Never kill jobs you did not start.
# Design: reports/llm_research.md sec. 4; build + smoke results: reports/llm_build.md.
#
# Usage (each invocation runs its arms sequentially in the foreground; run lanes detached, PIDs -> logs/RUNNING.txt):
#   bash nn_x/llm/run_matched.sh <GPU> <arm> [<arm> ...]
#   arms: probe  B23 B24  L23 L24  R23  E1  compare  [R23_E1 only if compare says PENDING; later: L2_23 L2_24 (only
#         if L1 passes), P4B (4B probe, exception)]
#         large models (reports/llm_large.md): P4 P9 (size probes, first)  Q4_23 Q4_24  Q9_23 Q9_24  compare_q4
#         compare_q9  curve (size curve 0.6B / 4B / 9B; CPU)
#   e.g. cd <repo>/nn_x/llm && mkdir -p logs && \
#        bash run_matched.sh 6 probe        # first, alone: prints PROBE GATE PASS/FAIL and the I3 decision
#        setsid nohup bash run_matched.sh 6 L23 L24 R23 E1 > logs/lane_gpu6.out 2>&1 & echo "$! lane6" >> logs/RUNNING.txt
#        setsid nohup bash run_matched.sh 7 B23 B24 > logs/lane_gpu7.out 2>&1 & echo "$! lane7" >> logs/RUNNING.txt
#   and when all arms (incl. E1) are done:  bash run_matched.sh <gpu> compare   (CPU only; any gpu arg)
#   Every training arm (B, L, R, E1 runs, L2, P4B) refuses to start until the probe has run AND passed the probe gate
#   (>= 25K padded tok/s, <= 30 GB peak; compare.py --i3_flag exits 3 on a failed gate). Re-running the probe on a
#   quieter lane replaces probe/ (and the I3 decision).
#   6 GPU-h HARD CAP: every GPU arm is recorded in logs/gpu_ledger.tsv (start/end, gpu, arm, lane pid); before each GPU
#   arm the lane refuses to start once the recorded lane wall-clock hours of ALL lanes reach CAP_GPU_H (a running arm
#   is never stopped, so one arm can overshoot). E1's two 4-epoch runs only count as a pair, so when E1 says extend
#   they start only if used + 2 x (L23 + L24 lane-hours) <= CAP_GPU_H; otherwise neither starts, compare refuses, and
#   the lead decides. Wall-clock on a shared lane >= GPU time, so the cap binds early. Only the lead raises CAP_GPU_H.
#   (Measured on GPU N, 2026-11-12 03:23 UTC: the probe ran at 27.6K padded tok/s, so an L arm is ~70 min of lane time,
#   not the ~16 min exclusive estimate below; base arms ~5.5 lane-hours, so an E1 extension will NOT fit under 6.)
#
# PRE-LAUNCH AMENDMENTS (reports/LLM_FEASIBILITY.md 5.2; declared before any GPU look):
#   S1  frozen sink token at position 0 = the checkpoint's own <|im_start|> row (id 151644; the random-init control
#       uses the same row of its own init). Layout: sink 0, user 1, history 2..H+1, every candidate H+2. Default in
#       model_llm/train_llm (no flag needed; --no_sink exists for ablation only and is never used here).
#       DEVIATION from the declared row <|endoftext|> (151643): on CPU at init the token after <|endoftext|> is treated
#       as a document start and still becomes a massive-activation sink (S1 acceptance FAILS: layer-13 user cosine
#       0.99999); <|im_start|> passes (reports/llm_build.md "Amendments"). Revert: add --sink_token endoftext to L_FLAGS.
#   E1  arm E1 (after L23 and L24): compare.py --e1_decide reads ONLY their holdout lines; if for either seed the
#       holdout-best evaluation is the final one and beats the previous evaluation, BOTH seeds are re-run once with
#       --epochs 4 (llm06_L1_e4_s23/24, on this lane) and A1/A2 are judged on those. Decided once, never re-decided;
#       compare refuses to run before it. R23 gets the same guard only if the P4B exception would otherwise fire:
#       compare then prints PENDING -> run arm R23_E1 (llm06rand_L1_e4_s23, 4 epochs) and re-run compare.
#   I3  the probe scores the holdout with bf16 and with fp32 trunk weights (train_llm.py --precision_check). If
#       |dNDCG| > 1e-4, every L/R/E1/L2/P4B arm writes its final holdout/val/test predictions in fp32 (--final_fp32,
#       added automatically from probe/i3_precision.json).
#   V0  every L/R arm runs with --no_val_log: the trainer never scores or logs val; compare.py reads val first.
#
# LARGE-MODEL ARMS (reports/llm_large.md, pre-declared 2026-11-12 before any large-model GPU run; the user asked for
#   experiments on models larger than ~1B): Qwen3.5-4B (Q4_23, Q4_24) and Qwen3.5-9B (Q9_23, Q9_24), the exact
#   qwen35_branch trunk, layout L1, the L flags otherwise unchanged (LoRA r=32 on all linear incl. in_proj_qkv/in_proj_z/
#   out_proj, projectors, fp32 score head, K9 aux, listwise loss, 2 epochs on the identical subsets, holdout every 200
#   steps and holdout-best checkpoint, S1 sink = that checkpoint's <|im_start|> row, --no_val_log, --final_fp32 per I3).
#   Batching: FeedStore.token_batches_left (same 56K budget counted in the branch layout's own padded rows; seed 23:
#   757 steps/epoch, 196.3 feeds/step vs 758 / 196.0 for the L arms) because token_batches' rows are 1.64x the budget
#   in the left-padded layout. Gradient accumulation (exact) only if the probe shows memory forces it (Q4_ACCUM/Q9_ACCUM).
#   Probe arms P4 / P9 (40 steps + one holdout eval + fp32-inference timing on 2000 holdout feeds): gate peak reserved
#   memory <= [n] GB AND estimated arm wall-clock <= 10 h on the probe lane (compare.py --probe_large; Q arms refuse
#   otherwise). No E1 extension for Q arms (holdout traces reported as information); acceptance = the A1/A2 below vs
#   the same B23/B24, computed by the same compare.py code (arms compare_q4 / compare_q9); size curve = arm curve.
#   Every arm launched by these lanes waits for a slot first: < 4 compute processes on the GPU and used + need <= [n] GB.
#
# ARMS (all train-only; checkpoints chosen on the fixed 30K random-like train holdout; val is never used to select):
#   B<s>  ranker baseline: UNMODIFIED nn/train.py, F7 stage-1 flags + --train_feeds 150000 --epochs 20 --eval_every 250,
#         holdout-selected; outputs redirected to nn_x/llm/baseline/ by run_nn_baseline.py (nothing written to nn/, preds/)
#   L<s>  Qwen3-0.6B pretrained, soft tokens, L1 layout, LoRA r=32 all linear (B=0 init) + RMSNorm gains + projectors +
#         fp32 score head + K9 aux head; listwise softmax per feed (equal feed weight) + 0.5(1-p_random) K9 tau 10;
#         AdamW LoRA lr 2e-4 / projectors+heads 1e-3, warmup 100, cosine to 10%, 56K-token batches, 2 epochs
#         (1516 steps for seed 23 with the sink token), holdout every 200 steps, holdout-best checkpoint kept; S1 sink
#         token; no val (V0).
#   R23   control: the same Qwen3-0.6B architecture, RANDOM init, fully trained (trunk lr 5e-4, warmup 200), seed 23.
#   Subset per seed = np.sort(default_rng(seed).choice(3151208, 150000)) minus the fixed 30K holdout
#         (= nn/train.py --train_feeds): seed 23 -> 148,600 feeds, seed 24 -> 148,557 feeds (verified on CPU).
#
# PRE-DECLARED ACCEPTANCE (copied from the research design; computed by compare.py, decision printed by it):
#   A1 (beats the ranker): pooled paired train-only val L-B > 2 SE, both seeds positive, holdout >= 0,
#      no slice below -2 SE (val: M<50, M>=50, Tue, Thu, all-cold-positive; holdout: late 20%).
#   A2 (useful ensemble member): (i) L within 0.002 of B on val AND on holdout (L >= B - 0.002), and
#      (ii) mean over seeds of eq(L_s,B_s) minus eq(B23,B24) > 0 on val with z >= 1.5 and holdout >= 0.
#      Also reported: rho(L,B) and the law-predicted gain 0.0186(1-rho) - 0.378|gap|.
# DECISION RULES:
#   Both fail: STOP. One exception: if L >= B - 1 SE and L beats the random-init control by > 2 SE, run ONE
#      Qwen3-4B-Instruct-2507 probe (arm P4B; apply A1/A2 to it; stop if it fails).
#   A1 passes: a 600K-feed slope check first (seed 23); continue only if the advantage keeps >= 50%.
#   A2 passes, or the slope check passes: full-data 0.6B members (1 epoch), then gates (a)+(b) and the frozen pool
#      protocol (candidate pools with/without the LLM family chosen on the holdout; replace a pick only with
#      val > 2 SE, no slice < -2 SE, src/presub_gate.py PASS).
#   Variant L2 (candidates also attend to each other) only if L1 passes. Qwen3.5 only via qwen35_branch.py.
#   Fallback (pre-declared, at most once, only if an L run diverges): --lora_r 64 --lr_lora 1e-4. No search on val.
#
# ESTIMATES (nn_x/llm/estimate.py; exact token counts; MFU is an ASSUMPTION, no GPU was used to measure it):
#   tokens (with the S1 sink token): train subset s23 39.36M real / 41.68M padded per epoch (padding eff 0.944,
#           758 steps/epoch at 56K tokens); holdout 10.0M, val 3.9M, test 7.9M real tokens.
#   L arm (0.6B, 6NT, 15% MFU of the vendor's peak rate on an EXCLUSIVE GPU): train 11 min + 8 evals (holdout only, V0)
#           ~3.8 min + final predictions 1 min = ~16 min per seed (pessimistic 2x: ~32 min).  Peak memory ~15 GB.
#           If I3 demands fp32 final predictions: + ~5-10 min per arm (fp32 CUDA cores, TF32 off; not measured).
#   R arm (8NT, all weights + AdamW state fp32): ~21 min (pessimistic ~42).  Peak memory ~20 GB.
#   B arm (ranker, 8K steps + 32 evals + full-split SplitData load): ~20-30 min per seed; ~40-45 GB (whole train split
#           is loaded on the GPU by the unmodified nn/train.py).
#   probe: 40 steps + one eval + the I3 fp32 holdout pass, ~5-10 min; prints measured tokens/s, peak memory and the
#           I3 decision -> check the MFU assumption first.
#   E1 (if it fires): two more L runs at 4 epochs, ~2 x 30 min exclusive (~2 h pessimistic).
#   Stage A total ~1.8 GPU-h exclusive (~3.5 h pessimistic). some GPUs currently run at 75-95% utilisation, so a
#   shared lane can be 2-3x slower than these exclusive figures.
# =====================================================================================================================
set -euo pipefail
gpu=${1:?usage: run_matched.sh <GPU> <arm>...}; shift
case "$gpu" in 0|1|2|3|4|5|6|7) ;; *) echo "GPU must be one id 0-7"; exit 1;; esac
HERE=<repo>/nn_x/llm
cd "$HERE"
mkdir -p logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # CUDA index == nvidia-smi index (need_mem queries -i $gpu); review fix
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 POLARS_MAX_THREADS=8
PY_LLM=<repo>/.venv_llm/bin/python   # transformers/peft env (reuses .venv's torch via a .pth)
PY_NN=<repo>/.venv/bin/python        # project env for the unmodified nn/train.py

need_mem() {  # refuse to start if the GPU lacks free memory (never touch other jobs)
  local need=$1 free
  free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$gpu" | awk -F', ' '{print int(($1-$2)/1024)}')
  if (( free < need )); then echo "GPU $gpu has ${free} GB free < ${need} GB needed; not starting"; exit 2; fi
}

B_FLAGS="--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
--select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --sched wsd --cooldown_frac 0.2 \
--aux_anneal 0.1 --retr_m 16 --attn_gate --lr 2e-3 --cooldown_mix 0.1 --train_feeds 150000 --epochs 20 --eval_every 250"
L_FLAGS="--backbone qwen3-0.6b --layout L1 --train_feeds 150000 --holdout 30000 --xt cvnd_pre2s --epochs 2 \
--token_budget 56000 --lr_lora 2e-4 --lr_proj 1e-3 --aux_pos 0.5 --aux_tau 10 --eval_every 200 \
--device cuda --dtype bf16 --grad_ckpt --threads 12 --no_val_log"   # --no_val_log = amendment V0
L_E4_FLAGS=${L_FLAGS/--epochs 2/--epochs 4}                            # amendment E1 (single 4-epoch re-run)
[[ "$L_E4_FLAGS" == *"--epochs 4"* ]] || { echo "internal: E1 flags"; exit 1; }
# ---- large-model arms (reports/llm_large.md): only the backbone changes (+ exact gradient accumulation if forced)
Q4_ACCUM=1   # set > 1 only if the P4 probe fails its memory gate (then re-probe); the effective batch stays 56K tokens
Q9_ACCUM=1
qflags() { local z=$1 acc; acc=$([[ $z == 4 ]] && echo $Q4_ACCUM || echo $Q9_ACCUM)
  echo "${L_FLAGS/--backbone qwen3-0.6b/--backbone qwen3.5-${z}b} --grad_accum $acc"; }
[[ "$(qflags 9)" == *"--backbone qwen3.5-9b"*"--no_val_log --grad_accum"* ]] || { echo "internal: Q flags"; exit 1; }
MAX_PROCS=4 MAX_GPU_GB=N   # GPU rules: at most 4 training processes and about [n] GB per GPU
gpu_used_gb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | awk '{print int($1/1024)}'; }
gpu_procs() { local uuid; uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid" || true; }
wait_slot() {  # wait (never displace anyone) until < MAX_PROCS compute processes and used + need <= MAX_GPU_GB
  local need=$1 n=0 used procs
  while :; do
    used=$(gpu_used_gb); procs=$(gpu_procs)
    if (( procs < MAX_PROCS && used + need <= MAX_GPU_GB )); then return 0; fi
    if (( n % 15 == 0 )); then echo "[$(date -u +%H:%M:%S)] GPU $gpu: $procs procs, ${used} GB used; waiting for a slot" \
      "(< $MAX_PROCS procs and used + ${need} <= $MAX_GPU_GB GB)"; fi
    n=$((n + 1))
    if (( n > 720 )); then echo "no slot on GPU $gpu after 24 h; giving up"; exit 2; fi
    sleep 120
  done; }
arm_need() { case "$1" in  # GB of GPU memory an arm needs (for wait_slot)
    B23|B24) echo 60 ;; L23|L24|L2_23|L2_24|E1|E1_23|E1_24|probe) echo 30 ;; R23|R23_E1) echo 35 ;; P4B) echo 60 ;;
    P4) echo 80 ;; P9) echo 140 ;;
    Q4_*) CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --probe_large "$HERE/probe_q4" 2>/dev/null || echo 80 ;;
    Q9_*) CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --probe_large "$HERE/probe_q9" 2>/dev/null || echo 140 ;;
    *) echo 30 ;; esac; }

i3_flag() {  # amendment I3: '--final_fp32' iff the probe found |dNDCG(bf16 vs fp32)| > 1e-4; refuse without a probe
  CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --i3_flag "$HERE/probe"; }  # exit 2: no probe; exit 3: probe gate FAILED

CAP_GPU_H=60                       # Stage A hard cap (LLM_FEASIBILITY 5.3), lane wall-clock hours summed over all lanes
LEDGER=$HERE/logs/gpu_ledger.tsv  # one line per GPU-arm start/end: kind, unix time, gpu, arm, lane pid
CUR_ARM=""
ledger() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$(date +%s)" "$gpu" "$2" "$$" >> "$LEDGER"; }
gpu_budget() {  # refuse to START GPU work once the ledger reaches CAP_GPU_H (exit 3); prints the hours used
  CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --gpu_hours "$LEDGER" --cap "$CAP_GPU_H"; }
trap 'if [[ -n "$CUR_ARM" ]]; then ledger end "$CUR_ARM"; fi' EXIT   # failed / interrupted arms are costed too

arm_B() { local s=$1 v=llmcmp_base_s$1
  [[ -f baseline/preds/holdout/nn_${v}_holdout.npy ]] && { echo "skip $v (done)"; return; }
  i3_flag > /dev/null; need_mem 60   # probe gate: no arm runs unless the probe passed
  $PY_NN run_nn_baseline.py --variant "$v" --seed "$s" $B_FLAGS > "logs/$v.out" 2>&1; }
arm_L() { local s=$1 v=llm06_L1_s$1 f
  [[ -f preds/${v}_trainonly_test.npy ]] && { echo "skip $v (done)"; return; }
  f=$(i3_flag); need_mem 30
  $PY_LLM train_llm.py --variant "$v" --seed "$s" $L_FLAGS $f > "logs/$v.out" 2>&1; }
arm_R() { local v=llm06rand_L1_s23 f
  [[ -f preds/${v}_trainonly_test.npy ]] && { echo "skip $v (done)"; return; }
  f=$(i3_flag); need_mem 35
  $PY_LLM train_llm.py --variant "$v" --seed 23 $L_FLAGS --random_init --lr_full 5e-4 $f > "logs/$v.out" 2>&1; }
arm_Le4() { local s=$1 v=llm06_L1_e4_s$1 f   # E1 extension run (only via arm E1)
  [[ -f preds/${v}_trainonly_test.npy ]] && { echo "skip $v (done)"; return; }
  f=$(i3_flag); need_mem 30   # the GPU cap for the pair was checked once by arm_E1 (--project_e1)
  $PY_LLM train_llm.py --variant "$v" --seed "$s" $L_E4_FLAGS $f > "logs/$v.out" 2>&1; }
arm_E1() {  # amendment E1: decide once from L23/L24 holdout lines; if extend, re-run BOTH seeds at 4 epochs here
  local out
  for v in llm06_L1_s23 llm06_L1_s24; do
    [[ -f preds/${v}_trainonly_test.npy ]] || { echo "E1: $v not finished; E1 runs after L23 and L24"; exit 1; }
  done
  out=$(CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --e1_decide --L llm06_L1_s23 llm06_L1_s24)
  echo "$out" | tee -a logs/e1_decision.out
  if [[ "$(echo "$out" | tail -1)" == "E1_EXTEND=1" ]]; then  # both 4-epoch runs must fit under the cap, or neither
    CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --gpu_hours "$LEDGER" --cap "$CAP_GPU_H" --project_e1
    arm_Le4 23; arm_Le4 24
  else echo "E1: no extension (judge the 2-epoch runs)"; fi; }
arm_E1s() {  # ONE seed of the E1 extension, so the pair can run on two GPUs in parallel (added 08:16 UTC, see
  local s=$1 f=results/e1_decision.json   # reports/llm_large.md); only after the decision (arm E1 / --e1_decide) said extend
  [[ -f $f ]] || { echo "E1_$s: no E1 decision yet (arm E1 or compare.py --e1_decide first)"; exit 1; }
  [[ "$(CUDA_VISIBLE_DEVICES="" $PY_LLM -c "import json; print(int(json.load(open('$f'))['extend']))")" == 1 ]] || {
    echo "E1_$s: the E1 decision says no extension; refusing"; exit 1; }
  CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --gpu_hours "$LEDGER" --cap "$CAP_GPU_H" --project_e1   # the pair must fit
  arm_Le4 "$s"; }
arm_R_E1() { local v=llm06rand_L1_e4_s23 f  # ONLY when compare printed PENDING (E1 guard on the control)
  [[ "$(CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --e1_R_needed)" == "1" ]] || {
    echo "R23_E1: not demanded by compare (results/e1_R_decision.json); refusing"; exit 1; }
  [[ -f preds/${v}_trainonly_test.npy ]] && { echo "skip $v (done)"; return; }
  f=$(i3_flag); need_mem 35
  $PY_LLM train_llm.py --variant "$v" --seed 23 $L_E4_FLAGS --random_init --lr_full 5e-4 $f > "logs/$v.out" 2>&1; }
arm_L2() { local s=$1 v=llm06_L2_s$1 f   # ONLY if L1 passed A1 or A2
  [[ -f preds/${v}_trainonly_test.npy ]] && { echo "skip $v (done)"; return; }
  f=$(i3_flag); need_mem 30
  $PY_LLM train_llm.py --variant "$v" --seed "$s" ${L_FLAGS/--layout L1/--layout L2} $f > "logs/$v.out" 2>&1; }
arm_P4B() { local v=llm4b_L1_s23 f       # ONLY under the pre-declared exception (see header)
  [[ -f preds/${v}_trainonly_test.npy ]] && { echo "skip $v (done)"; return; }
  f=$(i3_flag); need_mem 60
  $PY_LLM train_llm.py --variant "$v" --seed 23 ${L_FLAGS/qwen3-0.6b/qwen3-4b-instruct} $f > "logs/$v.out" 2>&1; }
arm_probe() {  # 40 training steps + one holdout eval + the I3 bf16-vs-fp32 holdout check, before the real runs
  if i3_flag > /dev/null 2>&1; then  # a PASSED probe is final: the I3 decision must not change between arms
    echo "skip probe (already passed; I3 decision fixed for Stage A; delete probe/ only with the lead's sign-off)"
    CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --probe "$HERE/probe"; return; fi
  need_mem 30
  $PY_LLM train_llm.py --variant llm06_probe --seed 23 $L_FLAGS --max_steps 40 --eval_every 40 --no_test \
      --precision_check --out "$HERE/probe" > logs/llm06_probe.out 2>&1
  grep -E "steps/epoch|step 40/40|BEST|I3 precision" logs/llm06_probe.out || true
  CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --probe "$HERE/probe" | tee logs/probe_gate.out; }
arm_P() {  # size probe P4 / P9: 40 steps + one holdout eval + fp32 timing -> probe_q<z>/probe.json and the size gate
  local z=$1 dir=$HERE/probe_q$1
  if CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --probe_large "$dir" > /dev/null 2>&1; then
    echo "skip P$z (already passed)"; CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --probe_large "$dir"; return; fi
  i3_flag > /dev/null   # the 0.6B probe (I3 decision) must exist and have passed
  $PY_LLM train_llm.py --variant llmq${z}_probe --seed 23 $(qflags "$z") --max_steps 40 --eval_every 40 --no_test \
      --probe_json "$dir/probe.json" --out "$dir" > "logs/llmq${z}_probe.out" 2>&1
  grep -E "LOAD REPORT|MISSING|steps/epoch|step 40/40|SIZE PROBE" "logs/llmq${z}_probe.out" || true
  CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --probe_large "$dir" 2>&1 | tee "logs/probe_q${z}_gate.out"; }
arm_Q() {  # large-model arm: refuses unless the 0.6B probe (I3) and this size's probe gate passed
  local z=$1 s=$2 v=llmq$1_L1_s$2 f need
  [[ -f preds/${v}_trainonly_test.npy ]] && { echo "skip $v (done)"; return; }
  f=$(i3_flag)
  need=$(CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --probe_large "$HERE/probe_q$z")   # exit 3: size gate FAILED
  need_mem "$need"   # allocator setting only (less fragmentation when sharing a GPU; no numerics change), 04:22 UTC
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY_LLM train_llm.py --variant "$v" --seed "$s" $(qflags "$z") $f > "logs/$v.out" 2>&1; }
arm_compare_q() {  # large-model arm sets: the same A1/A2 code vs the same B; no E1 (traces = information)
  local z=$1
  CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --L llmq${z}_L1_s23 llmq${z}_L1_s24 --B llmcmp_base_s23 llmcmp_base_s24 \
      --R llm06rand_L1_s23 --tag q$z --e1 info --size "Qwen3.5-${z}B" | tee "logs/compare_q$z.out"; }
arm_curve() { CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --curve 0.6B:stageA 4B:q4 9B:q9 --tag size_curve \
      | tee logs/size_curve.out; }
arm_compare() {  # refuses to run before the E1 decision; substitutes the 4-epoch runs if E1 extended
  CUDA_VISIBLE_DEVICES="" $PY_LLM compare.py --L llm06_L1_s23 llm06_L1_s24 --B llmcmp_base_s23 llmcmp_base_s24 \
      --R llm06rand_L1_s23 --tag stageA | tee logs/compare_stageA.out; }

for arm in "$@"; do
  echo "[$(date -u +%H:%M:%S)] GPU $gpu arm $arm start"
  if [[ "$arm" != compare* && "$arm" != curve ]]; then   # GPU arm: wait for a slot, then the lane-hour cap
    wait_slot "$(arm_need "$arm")"; gpu_budget; ledger start "$arm"; CUR_ARM=$arm; fi
  case "$arm" in
    probe) arm_probe ;;
    B23) arm_B 23 ;; B24) arm_B 24 ;;
    L23) arm_L 23 ;; L24) arm_L 24 ;;
    R23) arm_R ;;
    E1) arm_E1 ;;
    E1_23) arm_E1s 23 ;; E1_24) arm_E1s 24 ;;
    R23_E1) arm_R_E1 ;;
    L2_23) arm_L2 23 ;; L2_24) arm_L2 24 ;;
    P4B) arm_P4B ;;
    P4) arm_P 4 ;; P9) arm_P 9 ;;
    Q4_23) arm_Q 4 23 ;; Q4_24) arm_Q 4 24 ;; Q9_23) arm_Q 9 23 ;; Q9_24) arm_Q 9 24 ;;
    compare) arm_compare ;;
    compare_q4) arm_compare_q 4 ;; compare_q9) arm_compare_q 9 ;; curve) arm_curve ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  if [[ -n "$CUR_ARM" ]]; then ledger end "$CUR_ARM"; CUR_ARM=""; fi
  echo "[$(date -u +%H:%M:%S)] GPU $gpu arm $arm done"
done
