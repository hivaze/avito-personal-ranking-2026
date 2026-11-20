# Large soft-token LLM rankers (Qwen3.5-4B / 9B) vs the ranker on matched data: pre-declaration, runs, verdicts

Status: **PRE-DECLARED 2026-11-12 ~04:10 UTC, before any large-model probe or arm was launched and before any LLM-arm val number exists.** Results are appended below the line "RESULTS" only.
Why: the user asked for "better experiments on larger models, not 1b". This extends the pre-declared Stage A (reports/llm_build.md "Amendments", reports/LLM_FEASIBILITY.md §5) from Qwen3-0.6B to Qwen3.5-4B and Qwen3.5-9B, with the same inputs, loss, subsets, epochs, holdout selection, S1/I3/V0 amendments and the same A1/A2 acceptance vs the same ranker baselines B23/B24.
Code: `nn_x/llm/` only (`run_matched.sh`, `train_llm.py`, `data_llm.py`, `qwen35_branch.py`, `model_llm.py`, `compare.py`). Nothing in `nn/`, `src/`, `preds/`, `subs/`, `nn/ckpt` or `.venv` is touched; `.venv_llm` is unchanged (no packages installed).

## 1. Arms (all train-only; checkpoints selected on the fixed 30K random-like train holdout; val never selects anything)

| arm | variant | model | seed / subset | GPU lane |
|---|---|---|---|---|
| B23, B24 | llmcmp_base_s23/s24 | unmodified nn/train.py, F7 stage-1 flags, 20 epochs | 23 / 24 | GPU N, **done** 03:43-04:01 |
| L23, L24, R23, E1 | llm06_L1_s23/s24, llm06rand_L1_s23 | Qwen3-0.6B (as pre-declared in Stage A) | 23 / 24 | GPU N |
| P4 | llmq4_probe | Qwen3.5-4B, size probe | 23 | GPU N (exclusive at launch) |
| P9 | llmq9_probe | Qwen3.5-9B, size probe | 23 | GPU N (exclusive at launch), before the 0.6B lane |
| Q4_23, Q4_24 | llmq4_L1_s23/s24 | Qwen3.5-4B | 23 / 24 (148,600 / 148,557 feeds) | two different GPUs, chosen by load after P4 |
| Q9_23, Q9_24 | llmq9_L1_s23/s24 | Qwen3.5-9B | 23 / 24 | two different GPUs, chosen by load after P9 |

**Exact Q flags** (`run_matched.sh`, `qflags`): the L flags with only the backbone replaced, plus `--grad_accum`:
```
train_llm.py --variant llmq{4,9}_L1_s{23,24} --seed {23,24} --backbone qwen3.5-{4,9}b --layout L1 --train_feeds 150000 \
  --holdout 30000 --xt cvnd_pre2s --epochs 2 --token_budget 56000 --lr_lora 2e-4 --lr_proj 1e-3 --aux_pos 0.5 \
  --aux_tau 10 --eval_every 200 --device cuda --dtype bf16 --grad_ckpt --threads 12 --no_val_log --grad_accum 1 --final_fp32
```
- Trunk: the exact `qwen35_branch.Qwen35Ranker` (verified == HF forward: 2B 2.1e-6, 4B 4.0e-6 relative error on CPU). Layout L1. LoRA r=32, alpha 32, dropout 0.05 on q/k/v/o/gate/up/down + in_proj_qkv/in_proj_z/out_proj (all large linears; 4B: 200 modules, 60.9M LoRA params), all RMSNorm gains, projectors, fp32 score head + K9 aux head. Listwise softmax per feed with equal feed weight + 0.5(1-p_random) K9 at tau 10. AdamW (LoRA/norms 2e-4, projectors/heads 1e-3), warmup 100, cosine to 10%, clip 1.0, 2 epochs, holdout every 200 steps, holdout-best trainable state kept.
- S1: sink = the checkpoint's own `<|im_start|>` row (Qwen3.5 id 248045). Signed off by the lead for the 0.6B. The massive-activation analysis behind the row choice was done on Qwen3-0.6B only (disclosed, not re-screened).
- I3: `--final_fp32` on every Q arm, from the existing 0.6B probe decision (probe/i3_precision.json: dNDCG -0.000165 > 1e-4). Not re-decided per size.
- V0: `--no_val_log`; the trainer never scores val; compare.py is the first reader of val labels.
- Hyperparameters are **not** retuned for size (fixed in advance = the L-arm values).

**Implementation choices declared now (deviations from a literal copy of the L arms):**
1. **Batching for the left-padded branch layout.** In the branch trunk every row of a batch has `2 + max H + max M(rounded to 16)` tokens. With the L arms' `token_batches` (buckets by H+M only) those rows are **1.64x** the 56K budget on the seed-23 subset (68.4M rows per epoch vs 41.7M). The Q arms use `FeedStore.token_batches_left`: feeds ordered by (H bucket of 32, H = 256 its own bucket; M bucket of 16; random tie-break), filled greedily while B x row <= 56,000, batch order shuffled. This keeps the budget meaning (padded tokens of the trunk's own layout per batch) **and** the effective batch: seed 23 gives **757 steps/epoch and 196.3 feeds/step vs 758 and 196.0 for the L arms**, with 42.2M instead of 68.4M rows. Evaluation and prediction batching use the same function (prediction batching does not change scores beyond float noise; feed-alone == feed-in-batch was verified to 1e-5). Measured with `scratchpad/large/pad_infl.py` (CPU, light).
2. **Gradient accumulation** (`--grad_accum K`, exact: per-feed listwise terms are additive with the constant denominator; the K9 term is normalised by the full batch's feed-weight sum). Used **only** if a probe fails its memory gate; the effective batch stays the same 56K-token batch.
3. **No E1 extension for the Q arms** (a 4-epoch 9B pair would cost ~2x an arm). Each Q arm's holdout trace (best = final and final > previous?) is reported as information next to the verdict; a failing size that was still improving is flagged "possibly under-trained", but the verdict stands.
4. **P4B exception does not apply** to Q arms (it was the escalation from 0.6B to a 4B model). R (the 0.6B random-init control llm06rand_L1_s23) is reported in the size curve as information only.
5. **flash-linear-attention is NOT installed.** `qwen35_branch.branch_deltanet` calls the reference PyTorch `torch_chunk_gated_delta_rule` (vectorised, `solve_triangular`, <= 5 chunk iterations) and `F.conv1d` directly, which is exactly what the CPU parity tests verified. LLM_FEASIBILITY §5.4 asked for fla before any Qwen3.5 GPU run because GPU speed was unknown; the size probe below measures that speed directly, and its gate decides. No other kernel or numerics change.
6. **Slot rule in every lane:** before each GPU arm the lane waits (never displaces anyone) until the GPU has < 4 compute processes and used + need <= [n] GB (need = probe peak reserved + 10 GB for Q arms).

## 2. Size probes P4 / P9 (gate; run before any Q arm)
`run_matched.sh <gpu> P4` / `P9`: 40 training steps of the seed-23 arm (same flags) + one full holdout evaluation (bf16, 30K feeds) + an fp32-trunk inference timing on 2,000 holdout feeds (TF32 off, as the I3 final predictions) -> `probe_q{4,9}/probe.json`.
- **Estimated arm time** = 2 x setup + full steps (1,514 for seed 23) x steady s/step (mean of steps 6-40) + 8 holdout evals x measured eval time + fp32 model load + fp32 final predictions (holdout + val + test real tokens / measured fp32 tok/s).
- **Gate:** peak *reserved* memory (training and fp32 phase) **<= [n] GB** AND estimated arm time **<= 10 h** on the probe lane (an exclusive GPU at launch). `compare.py --probe_large` enforces it: Q arms of that size refuse to start otherwise.
- **If a size fails:** memory -> one retry with `--grad_accum 2` (then 4 if still over); time on a lane that turned out shared -> one re-probe on the least-loaded GPU. Otherwise the size is recorded as FAILED PROBE (no arms). The acceptance below never changes.

## 3. Acceptance (unchanged from Stage A; computed by the same compare.py code; per size, vs B23/B24)
- **A1 (beats the ranker):** pooled paired train-only val L-B > 2 SE; both seeds positive; holdout >= 0; no slice below -2 SE (val M<50, M>=50, Tue, Thu, all-cold-positive; holdout late 20%).
- **A2 (useful ensemble member):** (i) L >= B - 0.002 on val AND on holdout; (ii) mean over seeds of eq(L_s, B_s) - eq(B23, B24) > 0 on val with z >= 1.5, and holdout >= 0.
- Also reported: rho(L,B), rho(B23,B24), law-predicted eq gain 0.0186(1-rho) - 0.378|gap|.
- **Decision per size:** A1 pass -> 600K-feed slope check with that size (seed 23; continue only if the advantage keeps >= 50%). A2 pass -> full-data members of that size (1 epoch, 2 seeds), then gates (a)+(b) and the frozen pool protocol. Both fail -> STOP for that size.
- **If several sizes pass:** the next step is the full-data family of the **smallest** passing size (cheapest per member), unless a larger passing size has a higher A2(ii) val gain by more than 2 pooled SE (reported, information).
- The 0.6B arm set is judged exactly as pre-declared in Stage A (`compare`, E1 required, P4B exception text unchanged).
- Commands: `run_matched.sh <any> compare` (0.6B), `compare_q4`, `compare_q9` (`compare.py --e1 info --size ...`), then `curve`.

## 4. Size curve (pre-declared content; `compare.py --curve 0.6B:stageA 4B:q4 9B:q9 --tag size_curve`)
Per size: mean single train-only val and holdout NDCG of L and of B; paired L-B val and holdout (pooled over the 2 seeds, +-SE); eq(L,B)-eq(B,B) on val; rho(L,B) and rho(B23,B24); the law-predicted eq gain; A1/A2; plus the 0.6B random-init control R (val / holdout). Also reported (information): training wall-clock, s/step, peak memory per size.

## 5. Priors before any result [spec]
- P(A1) per size ~5-15%; P(A2) ~20-30%. Reason: the data has no text, project probes say the limit is information, not capacity (LLM_FEASIBILITY §2). Larger trunks add capacity and a stronger prior, not information.
- Expected holdout of an L arm: below B (0.304) at 2 epochs on 150K feeds for every size; the 0.6B probe was at 0.249 after 40 steps.

## 6. Compute
- Lane-hours ledger `nn_x/llm/logs/gpu_ledger.tsv`, CAP_GPU_H = 60 for 0.6B + 4B + 9B together (lead's decision). An arm never stops for the cap; new arms refuse to start once it is reached.
- GPU rules: the GPUs allowed; at most 4 training processes and ~[n] GB per GPU; lanes launched detached (`setsid nohup`), PIDs in `nn_x/llm/logs/RUNNING.txt`; own jobs stopped only by PID. CPU threads 12 per process. No NN training or inference on CPU.
- Pre-launch GPU smoke (04:03 UTC, scratch outputs only): Qwen3.5-4B, 256 smoke feeds, `--grad_accum 2 --probe_json`: runs end to end (load, left batching, accumulation, 2 holdout evals, fp32 rebuild + timing, probe JSON); 4B body 3,570M params, 64.8M trainable; peak reserved 48 GB. Qwen3-0.6B smoke with `--final_fp32` through the edited trainer: exit 0 (the L path keeps `token_batches`; the only change on it is the order of `loss.item()`).

---
RESULTS (appended after the pre-declaration; nothing above this line is edited after launch except typo fixes)

## R1. Launch record (2026-11-12 UTC)
**Probes (both on a GPU that was empty at launch; `nn_x/llm/probe_q{4,9}/probe.json`, logs `nn_x/llm/logs/llmq{4,9}_probe.out`):**

| probe | GPU | s/step | train tok/s (padded / real) | 8 holdout evals | fp32 final (real tok/s) | est. arm | peak reserved (alloc) | gate |
|---|---|---|---|---|---|---|---|---|
| P4 Qwen3.5-4B | N (empty) | 6.93 | 8,002 / 7,480 | 8 x 321 s = 0.71 h | 5,023 -> 1.21 h | **4.86 h** (train 2.92) | 111.4 GB (68.5) | **PASS** |
| P9 Qwen3.5-9B | N (empty) | 8.25 | 6,727 / 6,288 | 8 x 399 s = 0.89 h | 2,851 -> 2.13 h | **6.51 h** (train 3.47) | 141.5 GB (91.4) | **PASS** |

- Parameters: 4B body 3,570M, trainable 64.8M (LoRA 60.9M); 9B body 6,920M, trainable 86.2M (LoRA 80.2M); sink row id 248045 (RMS 0.0106 / 0.0123); emb RMS 0.01303 / 0.01347. The 9B loads 427/427 text weights with no missing or unexpected keys. Steps/epoch 757, feeds/step 196.3 (as declared).
- Holdout NDCG after the 40 probe steps (warmup lr, information only): 4B 0.26836, 9B 0.25290 (0.6B probe: 0.24909).
- **Bug found and fixed in the probe estimate (no effect on any training run):** `size_probe` counted evaluations with the probe's own `--eval_every 40` (38 evals) instead of the arm's 200 (8 evals, as declared in §2). Fixed in `train_llm.py` (`--arm_eval_every 200`); both JSONs recomputed by `nn_x/llm/fix_probe_json.py` (raw measurements kept; old totals 7.54 h / 9.83 h, which also passed). No gate outcome changed.
- No gate failed, so no grad accumulation (`Q4_ACCUM = Q9_ACCUM = 1`).
- **Allocator setting added to the Q arms** before they started: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (less fragmentation when sharing a GPU; reserved was 1.6x allocated in the probes). It does not change numerics.

**Lanes** (`nn_x/llm/lanes/*.sh`, detached with `setsid nohup`; PIDs in `nn_x/llm/logs/RUNNING.txt`). Load at launch: GPU N: 2 lead procs, [n] GB; GPU N: empty; GPU N: 2 procs, [n] GB; GPU N: 1 proc each, 59 GB (lead F12 members); GPU N: empty.

| arm | GPU | start (UTC) | why this GPU |
|---|---|---|---|
| P9, then L23 L24 R23 E1 (0.6B lane) | N | 04:06 / L23 04:23 | the task fixes the 0.6B lane on GPU N; P9 ran there first while it was empty |
| P4 | N | 04:05 | empty after B24 (04:01) |
| Q4_23 | N | 04:20 | 1 proc (59 GB); 59 + 122 GB need fits under the cap |
| Q4_24 | N | 04:20 | 1 proc (59 GB) |
| Q9_23 | N | 04:24 | empty; heaviest arm on the emptiest GPU |
| Q9_24 | N | 04:24 | shares only with my own 0.6B lane ([n] GB need + ~25 GB), so no extra lead GPU gets a second job |

## R2. Operations log
- 04:58-07:00: the lead started r6/r7 screens on every GPU (2-3 processes per GPU). Time-slicing gave my arms 20-50% of the SMs (`nvidia-smi pmon`): Q9_24 (GPU N, next to the 0.6B lane) ran at ~18-24 s/step and Q9_23 (GPU N) at ~18 s/step vs 8.25 in the probe; Q4 at ~12-15 s/step vs 6.93. Allowed sharing, so nothing was changed.
- 08:14: L23 (lane 1.84 h, holdout-best step 1400) and L24 (2.00 h, best step 1512) done. GPUs N had become empty. **R23 had just started on GPU N (loading, 49 s); I stopped it by PID (3126165, my own) and re-ran it on the empty GPU N** so that GPU N is left to Q9_24. The killed attempt is costed in the ledger (0.01 h); its log is kept as `logs/llm06rand_L1_s23.killed_0814.{out,log}`.
- 08:15: **E1 decided once** (`compare.py --e1_decide`, holdout lines only; `logs/e1_decision.out`, `results/e1_decision.json`): L23 best 1400 / final 1516, final 0.29890 < previous 0.29952 -> not improving; **L24 best = final 1512, 0.29928 > 0.29844 -> still improving -> EXTEND**: both seeds are re-run once at 4 epochs (llm06_L1_e4_s23/s24) and the 0.6B A1/A2 verdict is judged on those, as pre-declared.
- 08:15: to finish sooner, the E1 pair runs **in parallel** (new arms `E1_23` on GPU N and `E1_24` on GPU N; same `arm_Le4` command and flags as arm E1; they refuse unless the decision file says extend, and the pair cap projection `--project_e1` is checked: 20.3 + 7.7 <= 60 lane-hours). Sequential vs parallel does not change any number.

## R3. Qwen3.5-4B verdict (compare_q4, 10:00-10:40 UTC): **A1 FAIL, A2 FAIL -> STOP for 4B**
Command: `bash nn_x/llm/run_matched.sh 6 compare_q4` (= `compare.py --L llmq4_L1_s23 llmq4_L1_s24 --B llmcmp_base_s23 llmcmp_base_s24 --R llm06rand_L1_s23 --tag q4 --e1 info --size Qwen3.5-4B`); output `nn_x/llm/results/compare_q4.{txt,json}`, `logs/compare_q4.out`. A first run at 10:00 without `--R` (R23 was still training) gave identical A1/A2 lines (`logs/compare_q4_noR.out`).

| | val | holdout |
|---|---|---|
| Q4 s23 / s24 (single, train-only) | 0.37637 / 0.37409 | 0.29921 / 0.29893 |
| B23 / B24 | 0.37922 / 0.38083 | 0.30448 / 0.30325 |
| paired Q4-B, pooled | **-0.00480 +- 0.00081** (seeds -0.00285, -0.00674) | **-0.00479 +- 0.00049** |
| eq(Q4,B) - eq(B23,B24) | **-0.00224 +- 0.00056** | -0.00233 +- 0.00031 |
| rho(Q4,B) / rho(B23,B24) | 0.932, 0.939 / 0.946 | |
| law-predicted eq gain | -0.00062 | |
| R (0.6B random init, s23) | 0.37599 | 0.29978 |
| Q4 s23 - R | +0.00038 +- 0.00067 | -0.00056 +- 0.00038 |

- Slices (val, paired Q4-B): M<50 -0.00586, M>=50 -0.00406, Tue -0.00533, Thu -0.00422 (all below -2 SE); **all-cold-positive +0.01331 +- 0.00160** (information: the LLM arm is much better on feeds whose positives are all cold items, but overall it is 0.0048 weaker and the equal-weight pair loses 0.0022). Holdout late 20% -0.00352 +- 0.00111.
- A2(i) fails (gap 0.0048 > 0.002 on val and holdout); A2(ii) fails (eq gain negative at z -4).
- Both seeds were still improving at their last holdout evaluation (best = final step). Pre-declared: information only, no extension.
- Control (information): the pretrained 4B is not better than the 0.6B architecture trained from random init (+0.0004 +- 0.0007 val, -0.0006 +- 0.0004 holdout), so language pretraining adds nothing measurable here.
- Cost: train 274.5 / 280.3 min wall-clock per arm (pure training 220 / 227 min), on GPUs shared with lead jobs; lane-hours ~5.6 per arm.

## R4. Qwen3.5-9B verdict (compare_q9, 12:53 and 13:45 UTC): **A1 FAIL, A2 FAIL -> STOP for 9B**
Command: `bash nn_x/llm/run_matched.sh 6 compare_q9` (`compare.py --L llmq9_L1_s23 llmq9_L1_s24 ... --tag q9 --e1 info --size Qwen3.5-9B`); output `nn_x/llm/results/compare_q9.{txt,json}`. First run at 12:53 (val + holdout preds of both seeds existed; Q9_24 was still writing test preds); re-run at 13:45 after Q9_24 finished, with identical numbers and Q9_24's holdout trace added.

| | val | holdout |
|---|---|---|
| Q9 s23 / s24 | 0.37580 / 0.37361 | 0.29920 / 0.29898 |
| paired Q9-B, pooled | **-0.00532 +- 0.00080** (seeds -0.00342, -0.00722) | **-0.00477 +- 0.00050** |
| eq(Q9,B) - eq(B23,B24) | **-0.00253 +- 0.00056** | -0.00242 +- 0.00032 |
| rho(Q9,B) / rho(B23,B24) | 0.937, 0.934 / 0.946 | |
| law-predicted eq gain | -0.00081 | |
| Q9 s23 - R | -0.00019 +- 0.00065 | -0.00057 +- 0.00036 |
- Slices (val): M<50 -0.00727, M>=50 -0.00397, Tue -0.00539, Thu -0.00525 (all < -2 SE); all-cold-positive +0.01209 +- 0.00154. Holdout late 20% -0.00409.
- Holdout traces (information): s23 best = final step 1514 (still improving); s24 best step 1400 of 1512 (not improving).

## R5. Qwen3-0.6B Stage A verdict (compare, 13:45 UTC; judged on the E1 4-epoch runs as pre-declared): **A1 FAIL, A2 FAIL, no exception -> STOP**
Command: `bash nn_x/llm/run_matched.sh 6 compare` (unchanged Stage A arm, E1 required); `nn_x/llm/results/compare_stageA.{txt,json}`, `logs/compare_stageA.out`.
- Judged: llm06_L1_e4_s23 / s24 (4 epochs, holdout-best steps 2200/3032 and 2000/3024): val 0.37645 / 0.37604, holdout 0.29976 / 0.30000.
- Paired L-B val **-0.00378 +- 0.00077** (seeds -0.00277, -0.00480), holdout **-0.00398 +- 0.00046**; eq(L,B)-eq(B,B) val **-0.00203 +- 0.00055**, holdout -0.00233 +- 0.00030; rho(L,B) 0.936 / 0.943; law -0.00030.
- Control: L(e4, s23) - R val +0.00046 +- 0.00074, holdout -0.00001 +- 0.00042 -> "no evidence that language pretraining helps". The P4B exception needs L >= B - 1 SE (-0.00378 vs -0.00077): not met, so no R23_E1 guard and no P4B.
- Information: the 2-epoch runs (compare tag `q06_2ep`, `--e1 info`): val -0.00451 +- 0.00079, holdout -0.00454 +- 0.00047, eq -0.00242 +- 0.00055. Four epochs gained +0.00072 +- 0.00047 val, +0.00055 +- 0.00027 holdout, still -0.004 short of B.

## R6. Size curve (pre-declared §4; `compare.py --curve 0.6B-2ep:q06_2ep 0.6B-4ep:stageA 4B:q4 9B:q9 --tag size_curve`; `nn_x/llm/results/size_curve.{txt,json}`)

| size (trunk body) | L val | L hold | B val | B hold | L-B val | L-B hold | eq(L,B)-eq(B,B) val | rho(L,B) | rho(B,B) | law eq gain | A1/A2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B, 2 ep (0.44B) | 0.37552 | 0.29933 | 0.38003 | 0.30386 | -0.00451 +- 0.00079 | -0.00454 +- 0.00047 | -0.00242 +- 0.00055 | 0.941 | 0.946 | -0.00061 | F/F |
| Qwen3-0.6B, 4 ep (E1, judged) | 0.37624 | 0.29988 | 0.38003 | 0.30386 | -0.00378 +- 0.00077 | -0.00398 +- 0.00046 | -0.00203 +- 0.00055 | 0.940 | 0.946 | -0.00030 | F/F |
| Qwen3.5-4B, 2 ep (3.57B) | 0.37523 | 0.29907 | 0.38003 | 0.30386 | -0.00480 +- 0.00081 | -0.00479 +- 0.00049 | -0.00224 +- 0.00056 | 0.936 | 0.946 | -0.00062 | F/F |
| Qwen3.5-9B, 2 ep (6.92B) | 0.37470 | 0.29909 | 0.38003 | 0.30386 | -0.00532 +- 0.00080 | -0.00477 +- 0.00050 | -0.00253 +- 0.00056 | 0.936 | 0.946 | -0.00081 | F/F |
| random-init control (0.6B architecture, all 442M weights trained, 2 ep, s23) | 0.37599 | 0.29978 | | | | | | | | | |

**Extras (information only; `nn_x/llm/size_extra.py`, `logs/size_extra.out`, `logs/allcold_info.out`):**
- Paired differences between sizes at 2 epochs (same subsets, pooled over seeds): 4B - 0.6B val -0.00029 +- 0.00044, holdout -0.00026 +- 0.00026; 9B - 0.6B val -0.00081 +- 0.00047, holdout -0.00024 +- 0.00028; 9B - 4B val -0.00052 +- 0.00043, holdout +0.00002 +- 0.00024. **The curve is flat (if anything slightly down) over a 16x range of trunk size.**
- The pretrained trunks are no better than the same 0.6B architecture trained from random init (L-R within +-1 SE at every size).
- Prediction correlation on val: every LLM arm vs every other LLM arm (any size, pretrained or random init) 0.968-0.994 (Q4_23 vs Q9_23 0.9936; L06_23 vs R23 0.9896). B23 vs B24: 0.946. LLM vs B: 0.923-0.944. The trunk's size and pretraining barely change the function learned; the soft-token transformer over the ranker's inputs is one model class, weaker than the ranker by ~0.004-0.005 at matched data.
- The all-cold-positive val slice (3,422 of 14,423 feeds, 23.7%) favours every transformer arm, the random-init control included: s23 minus B23 = +0.0127 (R), +0.0122 (0.6B), +0.0132 (4B), +0.0128 (9B), all +-0.002, versus -0.008 on the other feeds. This is an architecture effect, not pretraining. The slice is defined by the labels (all positives cold), so it cannot be used as a test-time gate; the pre-declared equal-weight blend (A2(ii)) already loses 0.002.
- Cost per arm (wall-clock on GPUs shared with the lead's screens; `logs/<variant>.json`): 0.6B 89-101 min (4-ep 278-304), R 115 min, 4B 275-280 min, 9B 323-390 min; peak allocated 20 / 22 / 75 / 98 GB. Ledger total **46.7 of 60 lane-hours** (`compare.py --gpu_hours logs/gpu_ledger.tsv`).

## R7. Verdicts and recommendation (13:55 UTC)
| size | A1 | A2 | verdict |
|---|---|---|---|
| Qwen3-0.6B (Stage A, 4-epoch E1 runs) | FAIL (val -0.00378, z -4.9) | FAIL (gap 0.0038 > 0.002; eq -0.0020, z -3.7) | **STOP** (no P4B exception: L < B - 1 SE) |
| Qwen3.5-4B | FAIL (val -0.00480, z -5.9) | FAIL (gap 0.0048; eq -0.0022, z -4.0) | **STOP** |
| Qwen3.5-9B | FAIL (val -0.00532, z -6.6) | FAIL (gap 0.0053; eq -0.0025, z -4.5) | **STOP** |

**Recommended next step: stop the LLM-backbone route.** No full-data family and no slope check at any size. Add "LLM backbones as soft-token rankers (Qwen3-0.6B, Qwen3.5-4B, Qwen3.5-9B; LoRA; matched 150K; 2-4 epochs)" to the do-not list in COVERAGE_MAP / EXPERIMENT_REGISTRY.
Why this is final rather than "try longer / bigger":
1. **Size does nothing.** 0.6B -> 4B -> 9B is flat or slightly worse (9B - 0.6B -0.0008 +- 0.0005 val).
2. **Pretraining does nothing.** A random-init 0.6B ties every pretrained trunk.
3. **The arms are near-copies of each other** (rho 0.97-0.99) and less diverse from B than a second ranker seed is useful: every eq test is negative at z <= -3.7.
4. **Training longer helped by only 0.0007 val** (0.6B 2 -> 4 epochs). Some Q arms were still improving at their last evaluation, but closing a 0.004-0.005 gap would need ~6x that gain.
This matches the prior in LLM_FEASIBILITY §2: the data has no text, so an LLM adds capacity but no information.

## R8. Final section: size curve, verdicts, independent review, GPU hours, next step (14:15 UTC)
**Bottom line: larger trunks do not help.** The Qwen3.5-4B and Qwen3.5-9B soft-token rankers ran exactly as pre-declared: same inputs, loss, subsets, holdout selection, S1/I3/V0, and A1/A2 vs B23/B24. Each is about 0.005 below the ranker on val at matched data and fails A1 and A2, as the 0.6B does. An independent review reproduced every number from the labels and found no leak. None of its findings changes a verdict. The LLM-backbone route stops at every size tested.

### R8.1 Size curve
val | holdout. L = mean of the 2 seeds. Paired differences are pooled over seeds (+- SE). B (mean of B23/B24) = 0.38003 | 0.30386. R23 = the Qwen3-0.6B architecture trained from random init (2 epochs, seed 23) = 0.37599 | 0.29978.

| trunk (body params) | epochs | L val / hold | L-B val | L-B hold | eq(L,B)-eq(B23,B24) val | L(s23)-R23 val / hold | A1 / A2 |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B (0.44B) | 2 (information) | 0.37552 / 0.29933 | -0.00451 +- 0.00079 | -0.00454 +- 0.00047 | -0.00242 +- 0.00055 | -0.00007 +- 0.00060 / -0.00034 +- 0.00034 | F / F |
| Qwen3-0.6B (0.44B) | 4 (E1, judged) | 0.37624 / 0.29988 | -0.00378 +- 0.00077 | -0.00398 +- 0.00046 | -0.00203 +- 0.00055 | +0.00046 +- 0.00074 / -0.00001 +- 0.00042 | **F / F** |
| Qwen3.5-4B (3.57B) | N | 0.37523 / 0.29907 | -0.00480 +- 0.00081 | -0.00479 +- 0.00049 | -0.00224 +- 0.00056 | +0.00038 +- 0.00067 / -0.00056 +- 0.00038 | **F / F** |
| Qwen3.5-9B (6.92B) | N | 0.37470 / 0.29909 | -0.00532 +- 0.00080 | -0.00477 +- 0.00050 | -0.00253 +- 0.00056 | -0.00019 +- 0.00065 / -0.00057 +- 0.00036 | **F / F** |

- **Size (2 epochs, paired):** 4B-0.6B val -0.00029 +- 0.00044, holdout -0.00026 +- 0.00026. 9B-0.6B val -0.00081 +- 0.00047, holdout -0.00024 +- 0.00028. 9B-4B val -0.00052 +- 0.00043, holdout +0.00002 +- 0.00024. The curve is flat, if anything slightly down, over a 16x range of trunk size (0.44B to 6.92B body parameters).
- **Epochs (0.6B, 4 minus 2):** val +0.00072 +- 0.00047, holdout +0.00055 +- 0.00027.
- **Pretraining:** L - R is within about 1 SE at every size. Language pretraining adds nothing measurable.
- **Slices:** every val slice (M<50, M>=50, Tue, Thu) and both holdout slices (all, late 20%) are below -2 SE at every size.
- **Correlation:** rho(L,B) is 0.932-0.943, against rho(B23,B24) = 0.946. LLM arms correlate 0.968-0.994 with each other, R included. Law-predicted eq gain: -0.00030 (0.6B 4 ep), -0.00062 (4B), -0.00081 (9B).
- **Sources:** `nn_x/llm/results/compare_{stageA,q4,q9,q06_2ep}.{txt,json}`, `nn_x/llm/results/size_curve.{txt,json}`, `nn_x/llm/logs/size_extra.out`. Reproduced independently (R8.3).

### R8.2 Verdicts (pre-declared §3; unchanged from R7)
| size | A1 | A2 | decision |
|---|---|---|---|
| Qwen3-0.6B (judged on the E1 4-epoch runs) | FAIL (val -0.00378, z -4.9) | FAIL (gap 0.0038 > 0.002; eq -0.0020, z -3.7) | **STOP**. The P4B exception is not met: L-B = -0.00378 is below -1 SE = -0.00077. |
| Qwen3.5-4B | FAIL (val -0.00480, z -5.9) | FAIL (gap 0.0048; eq -0.0022, z -4.0) | **STOP** |
| Qwen3.5-9B | FAIL (val -0.00532, z -6.6) | FAIL (gap 0.0053; eq -0.0025, z -4.5) | **STOP** |

No size passes. So the "smallest passing size" rule, the 600K-feed slope check and the full-data family are not triggered.

### R8.3 Independent review: verdict CONFIRMED
The reviewer's scripts and outputs are copied to `nn_x/llm/review_large/`. The originals are in the session scratchpad.

**Reproduction.** The reviewer's own script loads the labels itself and scores with `src/metrics.ndcg_per_feed` (`nn_x/llm/review_large/recompute.py`, output in `recompute.{out,json}`).
- The holdout was re-derived with the nn/train.py logic (default_rng(12345), p_random > 0.9, M >= 50). holdout_y and holdout_meta are identical to the nn_x/llm, baseline and project files.
- The subsets have 148,600 (seed 23) and 148,557 (seed 24) feeds, with 0 overlap with the holdout.
- Every number in R3-R7 and R8.1 reproduces to the 5th decimal: single models, paired differences, eq, rho, law gain, slices, R controls and size differences. A1 = A2 = False for all four sets. rho(B23,B24) = 0.9463.
- Every LLM checkpoint's step equals the argmax of its holdout trace. All arms ran with `no_val_log` and `final_fp32`, and the training JSONs have no val keys.
- The E1 decision reproduces: L24 best = final step 1512, 0.29928 > 0.29844, so EXTEND. Ledger total 46.74 lane-hours.

**GPU tests.** Run on the trained `llmq4_L1_s23` and `llmq9_L1_s23`, on GPU N (0 processes at launch), in the foreground; they exited and left GPU N at 0 MiB. Peak memory 21 / 40 GB. Script `gpu_leak_test.py`, outputs `gpu_leak_q{4,9}.{out,json}`. Results 4B | 9B, **ALL PASS**:
- Permutation equivariance: 1.5e-6 | 9.5e-7 in fp32 (bf16: 1.0e-3 | 8.7e-4; score std 1.42).
- The model reads no label or position key. Flipped labels plus reversed positions change scores by exactly 0.0 (repeat-call noise floor 0).
- Cross-feed isolation: exactly 0.0. A feed scored alone vs in its batch differs by 3.3e-6 | 1.3e-6 in fp32.
- Other-candidate isolation under the L1 mask: exactly 0.0, while the perturbed candidate itself moves by 7.3 | 3.7. Positive control: under the L2 mask the other candidates move by 5.7e-3 | 1.0e-3, so the test can detect leakage.
- Branch trunk vs the HF `Qwen3_5TextModel` forward: max relative error 4.5e-6 | 1.6e-6 over 18 candidates. This is the first parity check for the 9B.
- The checkpoints reproduce the saved predictions: val 2.6e-6 | 3.8e-6, holdout 8.7e-4 | 9.0e-4.

**No other job was disturbed.**
- The only kill was my own R23 train_llm process (PID 3126165, parent lane 3335356). There was no pkill and no other signal.
- No OOM or traceback appears in the 540 lead logs written since 04:00.
- Polled memory peaked at [n] GB per GPU, under the [n] GB rule.
- Nothing was written to preds/, subs/, nn/ckpt or nn/logs.
- No LLM process remained at 14:09 UTC (ps check). Every lane in `nn_x/llm/logs/RUNNING.txt` is marked ENDED.

**Findings.** None changes a verdict. Each is listed with its effect:
1. **Unequal checkpoint selection favours B.**
   - B was selected among **39** holdout evaluations: eval_every 250 over 9,560 / 9,520 steps, checked in `nn_x/llm/baseline/logs/llmcmp_base_s{23,24}.log`. That is not the 32 stated in llm_build.md §3.5 (line 254). The L arms used 8 evaluations, and the E1 runs 16.
   - B's selected step 1250 is also its val maximum for both seeds (0.37922, 0.38083). Its holdout trace jumps by about 0.005 between neighbouring evaluations: s23 scores 0.30240 / 0.30448 / 0.29914 at steps 1000 / 1250 / 1500. So B's val carries some selection optimism, plausibly about 0.001.
   - Why the verdicts are robust: the val gaps of 0.0038-0.0053 would still exceed the 0.002 A2(i) bar after removing 0.001. A1 would still be z <= -3.6. In eq(L,B) - eq(B23,B24), B appears on both sides, so its inflation largely cancels.
   - This was pre-declared.
2. **Pre-declaration time.**
   - The header says ~04:10 UTC. The Write of this file was at **04:05:48 UTC**, 9 s before the P4 ledger start (04:05:57). That is long before the first LLM val prediction (05:56) and the first compare run (10:00).
   - The text above "RESULTS" is byte-identical to that Write (diff exit 0). Later changes are appends only (04:26, 08:15, 10:40, 13:53, and this section).
   - The A1/A2 code in compare.py was frozen at 04:01:44; only additions followed.
   - The header is left unedited (pre-declared lines are not edited) and the exact time is recorded here.
3. **Size-curve command.** R6 was run by hand with an extra `0.6B-2ep:q06_2ep` row. The pre-declared `curve` arm is `0.6B:stageA 4B:q4 9B:q9`. The extra row is information only, and the pre-declared rows are unchanged.
4. **Scope of R7's "why this is final".** It is stronger than the evidence.
   - What the evidence covers: one fixed LoRA recipe (no per-size retuning), 150K matched feeds, and 2 epochs (4 for the 0.6B).
   - What it does not cover: three of the four Q arms (Q4_23, Q4_24, Q9_23) were still improving at their last holdout evaluation, and no full-data run exists.
   - The pre-declared rule still says STOP. With +0.00072 +- 0.00047 val per doubling of epochs at 0.6B, closing a gap of about 0.005 is implausible, but it was not tested.
   - The do-not entry in R8.5 therefore names the recipe and the data scale.
5. **All-cold-positive slice.** The R6 numbers were seed-23-vs-B23 single comparisons. Pooled over seeds (val, paired): 0.6B 4 ep (judged) +0.00929 +- 0.00151; 0.6B 2 ep +0.01133 +- 0.00155; 4B +0.01331 +- 0.00160; 9B +0.01209 +- 0.00154. R23 is single-seed only: +0.0127. This remains information only. The slice is defined by the labels, so it cannot gate test predictions.
6. **fp32 final predictions depend slightly on batching.**
   - For 16 holdout feeds of the 4B (M 59-274, H 3-256), the saved predictions differ from a 16-feed re-score by up to 8.0e-4 absolute (median 9.5e-7). The same feed scored alone differs by up to 4.0e-4 (`gpu_holdout_repro_q4.out`). Score std is 1.42 and max |score| is 4.8.
   - The likely cause is rounding in the chunked delta rule when left padding shifts the chunk alignment.
   - The effect is negligible for NDCG, but it is larger than the "1e-5" stated in §1 item 1. That figure holds for the val feeds in test G3 (3.3e-6), not for these holdout feeds.
7. **Bookkeeping gaps.**
   - (a) The 04:03 pre-launch GPU smoke (Qwen3.5-4B on GPU N, about 1 min) is not in the ledger.
   - (b) V0 covers only L and R. B's val NDCG was logged by the unmodified nn/train.py at 03:52 and 04:01, before the pre-declaration. This has no effect, because acceptance is relative and B was fixed before any L existed.
   - (c) compare_q4 read R23's val at 10:40 while the E1 runs were still training. No code file was modified after 08:15, and the E1 flags were fixed at launch, so nothing could adapt to it.

### R8.4 GPU hours
Ledger `nn_x/llm/logs/gpu_ledger.tsv`, computed with `CUDA_VISIBLE_DEVICES="" .venv_llm/bin/python nn_x/llm/compare.py --gpu_hours nn_x/llm/logs/gpu_ledger.tsv --cap 60`.

| block | arms (lane-hours) | total |
|---|---|---|
| ranker baselines | B23 0.15, B24 0.15 | 0.30 |
| Qwen3-0.6B | probe 0.16, L23 1.84, L24 2.00, R23 2.41 (+0.01 killed attempt), E1_23 5.05, E1_24 5.62 | 17.09 |
| Qwen3.5-4B | P4 0.22, Q4_23 5.57, Q4_24 5.67 | 11.46 |
| Qwen3.5-9B | P9 0.29, Q9_23 8.33, Q9_24 9.27 | 17.89 |
| **total** | | **46.74 of 60** |

- These are wall-clock lane-hours on GPUs shared with the lead's screens. Time-slicing gave my arms 20-50% of the SMs (R2), so the hours overstate GPU time.
- Exclusive-GPU estimates from the probes were 4.86 h per 4B arm and 6.51 h per 9B arm. The measured lane-hours were 5.6 and 8.3-9.3 h.
- Not in the ledger: the 04:03 smoke (about 1 min) and the reviewer's GPU tests (GPU N, a few minutes around 14:00-14:06).

### R8.5 Next step
- **Stop the LLM-backbone route.** No 600K slope check, no full-data family, and no further sizes or epochs.
- **For the lead (I did not edit lead files):**
  - Add this entry to the do-not list in COVERAGE_MAP.md and EXPERIMENT_REGISTRY.md: "LLM backbones as soft-token rankers over the ranker's inputs. Qwen3-0.6B, Qwen3.5-4B and Qwen3.5-9B; LoRA r32 on all large linears with one fixed recipe (not retuned per size); layout L1; matched 150K-feed subsets; 2 epochs (4 for 0.6B). A1 and A2 FAIL at every size (L-B val -0.0038 to -0.0053; eq z <= -3.7). Neither size (16x) nor language pretraining adds anything measurable. Not tested: per-size tuning, more than 150K feeds, more than 2 epochs at 4B/9B."
  - CURRENT_STATUS §4 still lists the LLM large matched test as running. It has ended.
- **Reopening:** only for a reason this test did not cover, such as a new input that a language model could use. Not for a larger trunk or longer training: size and pretraining were flat here.
