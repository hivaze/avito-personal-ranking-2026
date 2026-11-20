# Soft-token LLM ranker: build and CPU smoke test (decisive GPU test ready, NOT run)

Status: built and smoke-tested on CPU, 2026-11-12. **The 4 pre-launch amendments (S1, E1, I3, V0) are implemented and the full CPU suite re-passes; one deviation (the S1 sink row) is explained in the "Amendments" section, which supersedes the older text below wherever they differ.** **A second-pass adversarial review (same section) re-ran everything, enforced the probe gate and the 6 GPU-h cap in `run_matched.sh`, and flags that at the probe's measured 27.6K tok/s an E1 extension does not fit under the cap.** **No GPU was used** (every command ran with `CUDA_VISIBLE_DEVICES=""`; the only GPU-adjacent call was a read-only `nvidia-smi` memory query). **Nothing was trained** beyond 1-6 optimiser steps on a few dozen feeds for the smoke tests. Nothing in `nn/`, `src/`, `pretrain/`, `data/`, `preds/`, `subs/`, `nn_x/solvers/` or `nn_x/r6/` was edited.
Design source: `reports/llm_research.md` (verdict **long shot**). This report covers only the build.

## TL;DR
1. **Isolated env `.venv_llm` works, and the project `.venv` is unchanged.**
   - transformers 5.22.2, peft 0.21.2 and accelerate 1.15.0 are installed in `.venv_llm` only.
   - It reuses `.venv`'s torch 2.11.0+cu130 through a `.pth` fallback.
   - Proof: an empty before/after `uv pip freeze` diff and zero files in `.venv` changed. One incident is disclosed in §1.
2. **Code is in `nn_x/llm/`.**
   - A read-only data layer that reproduces `nn/data.SplitData` bit for bit.
   - The Qwen3 soft-token ranker (layouts L1/L2, LoRA, fp32 score head + K9 aux head).
   - An exact Qwen3.5 "branch" DeltaNet trunk.
   - The trainer, a wrapper that runs the unmodified `nn/train.py` with redirected outputs, the A1/A2 evaluator, an exact token/FLOP estimator, and the launch script `run_matched.sh` (not executed).
3. **All CPU smoke tests pass** (Qwen3-0.6B on 32 real feeds, plus Qwen3.5-2B):
   - Candidate permutation equivariance: max error 8e-6.
   - A candidate scored in the batched L1 pass equals the same candidate scored alone as `[user, history, c]`: max error 1e-6.
   - No leakage: labels and positions are never read; perturbing them changes scores by 0.0; feed isolation is exact; in L1 other candidates have exactly zero influence.
   - One optimiser step lowers the loss, and the frozen weights do not change.
   - Qwen3.5 branch equals HF's own forward: relative error 9e-6. The naive causal append is order-dependent: scores change by up to 0.44.
4. **Compute** (exact token counts; MFU is assumed, not measured):
   - One L arm (Qwen3-0.6B, 2 epochs over 148.6K feeds): about 18 min on an exclusive GPU (36 min pessimistic), about 15 GB peak.
   - **Stage A in total (probe + 2 L + 1 R + 2 B + compare) is about 1.8 GPU-h exclusive, about 3.5 h pessimistic.**
   - some GPUs currently run at 75-95% utilisation, so expect wall-clock 2-3x longer on a shared lane.
5. **Next step**, when the lead frees a lane: `bash nn_x/llm/run_matched.sh <gpu> probe` (now also makes the I3 decision; see "Amendments" for the updated protocol).
   - The probe is 40 steps (about 5 min) and measures real tokens/s and memory on the GPU.
   - Then run the Stage A arms, then `compare`. The acceptance rules and decision tree are pre-declared in the script header and computed by `compare.py`.


## Amendments (pre-launch, 2026-11-12): S1, E1, I3, V0

All four amendments from `LLM_FEASIBILITY.md` §5.2 are in `nn_x/llm/`. The full CPU suite was re-run and **all checks pass**. Still nothing has run on a GPU, and `run_matched.sh` has not been executed.
- **No GPU:** every command ran with `CUDA_VISIBLE_DEVICES=""`, at most 12 threads.
- **`.venv` untouched:** `uv pip freeze` gives the same 94 lines before and after, and 0 files in `.venv` changed.
- **Edits:** only files under `nn_x/llm/` and this report.

**One deviation.** S1 uses the checkpoint's `<|im_start|>` row (id 151644) as the sink, not the declared `<|endoftext|>` row (151643).
- **Why:** the declared row fails S1's own pre-declared acceptance check (below).
- **How it was chosen:** on hidden-state geometry at initialisation only: no labels, no GPU, no training. (Review correction: the screening and T10 fed the soft tokens of *val* feeds, i.e. val inputs but never val labels or scores. That carries no label information, but "no val" was inaccurate.)
- **To revert:** add `--sink_token endoftext` to `L_FLAGS`.

| amendment | what changed | files |
|---|---|---|
| **S1** sink token | One frozen buffer `sink_emb` holds the checkpoint's own embedding row (the random-init control uses the same row of its own init). It is never trained and is saved in the checkpoint state. **Layout:** sink 0, user 1, history 2..H+1, every candidate H+2, and every real token may attend to the sink. Length bucketing counts the extra token. Qwen3.5's branch trunk gets the same prefix (row `<|im_start|>` = 248045). On by default; `--no_sink` is for ablation only | `model_llm.py`, `qwen35_branch.py`, `data_llm.py` (`seq_len`/`token_batches` `n_pre`), `train_llm.py`, `estimate.py` |
| **E1** under-training guard | `compare.py --e1_decide` reads **only** the `step` and `holdout_ndcg` of each evaluation in `logs/<L>.json`. **Rule:** extend if, for either seed, the holdout-best evaluation is the final one and beats the previous one. The decision is written once to `results/e1_decision.json` and never re-decided. **Arm `E1`** runs that decision and, if it says extend, re-runs **both** seeds at `--epochs 4` (`llm06_L1_e4_s23/24`). **`compare`** refuses to run before the decision and judges A1/A2 on the 4-epoch runs if E1 extended. **R guard:** only if the P4B exception would fire (A1 and A2 both fail, L ≥ B−1 SE, L−R > 2 SE) *and* R23 was still improving, compare prints `PENDING` and writes `results/e1_R_decision.json`. Then arm `R23_E1` (4 epochs; refuses unless demanded) runs and compare is re-run with the 4-epoch control. At most one extension | `compare.py`, `run_matched.sh` |
| **I3** inference precision | **Probe:** runs `--precision_check`, which scores the holdout twice: on the run's path (bf16 frozen weights + autocast), and with an **fp32-trunk copy** (re-loaded from the checkpoint plus the trained state; TF32 off). It writes `probe/i3_precision.json` with `final_fp32 = abs(ΔNDCG) > 1e-4`, plus the score correlation, the max abs score difference and within-feed tie counts. **Other arms:** every L/R/E1/L2/P4B arm reads that file through `compare.py --i3_flag` and adds `--final_fp32` when demanded. They **refuse to start if the probe has not run**. **Probe gate:** `compare.py --probe` prints it (≥ 25K padded tok/s, ≤ 30 GB) together with the I3 decision | `train_llm.py`, `model_llm.py` (`fp32_copy`), `compare.py`, `run_matched.sh` |
| **V0** val hygiene | `--no_val_log` is in `L_FLAGS`, so it applies to L, R, E1, L2, P4B and the probe. The trainer never computes val NDCG: neither at evaluation steps nor at the end. Log lines and the JSON history carry no val entries (`val_ndcg: null`). Val *predictions* are still written, so `compare.py` is the first reader of val labels. Evaluation time per arm drops from 5.3 to 3.8 min (estimate) | `train_llm.py`, `run_matched.sh`, `estimate.py` |

### S1: why the declared `<|endoftext|>` row was replaced
Massive activation = max |h| > 300 at layer 13; ordinary tokens there are about 20. The data are 24 fresh val feeds (276 user pairs), measured in smoke test T10 at initialisation.

| sink at position 0 | user tokens massive | user-token cosine across users, layer 13 (max / mean) | S1 acceptance (max < 0.99) |
|---|---|---|---|
| none (old layout) | 24/24 | 0.999999 / 0.99999 | fail |
| `<|endoftext|>` (declared) | **21/24** | **0.999998** / 0.911 | **fail** |
| **`<|im_start|>` (implemented)** | **0/24** | **0.970** / 0.844 | **pass** |

**Mechanism** [P]. Qwen3 treats the token after `<|endoftext|>` as the start of a new document, so it becomes a second sink.
- With *real* vocabulary rows in the user slot (12 feeds), `<|endoftext|>` still makes 5/12 of them massive.
- `<|im_start|>`, `<|im_end|>`, "The", "\n", "user", "Hello" and "," each make 0/12 massive.

**Why `<|im_start|>`** among the passing rows:
- It is the canonical first token of every Qwen chat sequence and is a special token.
- It was screened on 12 feeds and confirmed on 16 and then 24 fresh feeds (cosine max 0.936 / 0.945 / 0.970).
- Its embedding RMS is 0.0118, against 0.029 for an average row.

**Side finding** (not acted on, since it is outside the four amendments):
- At initialisation, a small, feed-dependent share of *history and candidate* soft tokens also enter the massive regime: 0/6,514 in T10, 25/3,815 and 46/3,489 in two other samples.
- Matching each soft token's per-dimension mean and std to the embedding matrix removed them (0/3,815) and kept the user non-massive with the `<|im_start|>` sink (cosine max 0.985).
- This is a one-line change to the ScaledLN initialisation, if the lead wants it before launch. Scripts: `…/scratchpad/llm/amend/proto_sink{2,3,4,5}.py`.

### Re-run CPU suite (Qwen3-0.6B with the sink unless stated; 32 real feeds; 12 threads; a high load average)

| check | result |
|---|---|
| T1 preset parity and subsets; T2 `FeedStore` == `SplitData` | PASS (unchanged: 148,600 / 148,557; bit-identical) |
| T3 build | PASS. 196 LoRA modules, LoRA B = 0, sink is a buffer (not a parameter), n_pre = 2 |
| T4 permutation equivariance within a feed | PASS. Max abs score difference: L1 1.1e-5, L2 9.5e-6 |
| T5 batched == single | PASS. `[sink,u,H,c]` alone vs batched 1.7e-6; feed alone vs in a padded batch 9.7e-6 |
| T6 leakage | PASS. y/pos never read; flipped labels Δ 0.0; **cross-feed Δ 0.0**; L1 other-candidate Δ 0.0 (L2: 64.2) |
| T7 optimiser and **tiny overfit set** | PASS. One step: 3.954 → 3.904; frozen base and sink unchanged. **8-feed overfit eval loss over 8 steps: 4.577 → 4.490 → 4.268 → 3.814 → 3.491 → 3.303 → 3.360 → 3.202 → 3.146** |
| T8 CPU throughput | info. Forward 0.81 TFLOP/s; training with checkpointing 0.39 TFLOP/s (6N) |
| T9 random-init control | PASS. 442.2M trainable; sink = its own init row (RMS 0.0203); `fp32_copy` exact (0.0); one step 3.948 → 3.858 |
| T10 S1 acceptance | PASS (table above) |
| T11 **HF native forward parity** (review R1) | PASS. Batched L1 score vs `model.lm(inputs_embeds=[sink,u,h_old..h_new,c])` with HF's own causal mask and positions: max abs difference **5.2e-6**. 12 checks, H = 256/56/71/0, non-zero LoRA B |
| T12 I3 path | PASS. `fp32_copy()` reproduces the trained model exactly (0.0). CPU stand-in (bf16 weights + CPU autocast vs fp32 trunk, 32 train feeds after 9 steps): ΔNDCG −0.0036 ± 0.0026, score corr 0.9997, max abs difference 0.58 at score std 2.4, 0 ties, so `final_fp32 = True` on CPU |
| Q1-Q4 Qwen3.5-2B branch with the sink | PASS. Naive append order-dependent (0.40); branch equivariance 6.6e-5; **branch == HF forward over `[sink,u,H,c]`: relative error 2.1e-6**; one step 4.298 → 4.285 |
| `smoke_baseline.py` (B-arm wrapper, unchanged) | PASS |
| `train_llm.py` end to end with a V0 spy (CPU bf16 emulation) | PASS. Run with `--no_val_log --precision_check`: **0 val-scoring calls**, no val keys in the history, `val_ndcg: null`; I3 JSON written. Positive control without `--no_val_log`: the spy sees 2 calls. Run with `--final_fp32`: holdout NDCG equals the I3 fp32 number of the matching run (0.183554) |
| `compare.py --probe / --i3_flag` | PASS. Reports the gate (correctly FAIL at CPU speed) and the I3 decision; prints `--final_fp32`; exits 2 when no probe exists |
| `compare.py` E1 / R-guard logic (stand-in predictions + synthetic holdout traces) | PASS. Refuses before the decision. No-extension case reproduces the earlier stand-in result exactly (+0.00167 ± 0.00077). The decision is never redone after logs change. Extension case refuses until the e4 runs exist, then judges them. R guard: PENDING, then `--e1_R_needed` = 1, then refuses without R e4, then judges with R e4 and fires the exception. A non-improving R decides directly |

Artefacts are in `/tmp/scratchpad/llm/amend/`: `smoke/`, `smoke35/`, `e2e/`, `cmp_tests/`, `baseline_test/`, `estimate_amended.json`, and the freeze files.

### Updated protocol and cost
```
cd <repo>/nn_x/llm && mkdir -p logs
bash run_matched.sh <GPU> probe                     # alone, first: PROBE GATE + I3 decision (probe/i3_precision.json)
setsid nohup bash run_matched.sh <gpuA> L23 L24 R23 E1 > logs/lane_gpuA.out 2>&1 &   # E1 may add 2 x 4-epoch L runs
setsid nohup bash run_matched.sh <gpuB> B23 B24        > logs/lane_gpuB.out 2>&1 &
bash run_matched.sh <any 4-7> compare                   # CPU; if it prints PENDING: run arm R23_E1, then compare again
```

**Tokens** (exact, with the sink):
- Seed 23: 39.36M real / 41.68M padded per epoch, 758 steps per epoch, so **1,516 steps** for 2 epochs (8 holdout evaluations).

**Time per arm** [inf, MFU assumed]:
- L arm: train 11.4 + evaluations 3.8 + final predictions 1.0 = **≈ 16 min** exclusive (≈ 32 min pessimistic).
- If I3 demands fp32: + ≈ 5-10 min per arm, because fp32 on CUDA cores with TF32 off is assumed, not measured. The probe itself grows by about 2-4 min.

**Two things to watch:**
- The CPU stand-in of I3 already exceeds the 1e-4 threshold. Be ready for `--final_fp32` on every arm.
- E1, if it fires, adds about 1 h exclusive (2 h pessimistic). The 6 GPU-h hard cap is unchanged.

### Adversarial review of the amendments (2026-11-12, ~03:20-03:50 UTC; CPU only, 12 threads, `.venv` freeze unchanged)

**Re-run (all pass; every reported number reproduced):**
- `smoke_test.py`: 12/12 PASS. T7 3.95359 → 3.90402 and the overfit trace, T10 0.969853 / 0.999998 / 0.999999, T11 5.2e-6, T12 dNDCG −0.00361 ± 0.00256 with `fp32_copy` error 0.0, and T9 all identical to the amendment run.
- `smoke_qwen35.py` Q1-Q4 PASS (Q3 2.1e-6). `smoke_baseline.py` PASS. B flags are identical across `run_matched.sh`, `smoke_baseline.py` and research §4.4.
- End to end (`train_llm.main` with a val spy, CPU bf16): `--no_val_log --precision_check` gives 0 val-scoring calls and `val_ndcg: null`. With `--final_fp32`, holdout NDCG 0.183554 = the I3 fp32 value.
- **compare.py vs an independent re-implementation of research §4.6.** Own data loading and `src/metrics.ndcg_per_feed`, five synthetic scenarios (A1 pass, STOP ×2, exception, slice failure): A1, A2 and exception decisions all match, with pooled deltas and SEs equal to 1e-9. The A1/A2 code is unchanged from the reviewed build.
- Script: `bash -n` passes. Its new gate and ledger logic was exercised by a harness that evaluates the script's own functions with every training command and nvidia-smi stubbed (13 scenarios). `run_matched.sh` itself was never executed by the review, and it is not auto-started (no cron, timer or caller).

**Verdict on the amendments.**
- **S1:** the deviation to `<|im_start|>` is justified by T10. Per layer, with `<|endoftext|>`, 21/24 users are massive from layer 6 on (max|h| ≈ 4,700), versus 0/24 with `<|im_start|>` at every layer.
- **E1, I3, V0:** implemented as declared.

**Fixes made by the review** (`compare.py`, `run_matched.sh`, `smoke_test.py`, this report):
1. **Probe gate enforced.** Before this fix the gate was only printed, so a lane like `probe L23 …` would train after a FAIL. `compare.py --i3_flag` now exits 3 on a failed gate, and every training arm (B included: "otherwise run no arms") calls it.
2. **6 GPU-h hard cap enforced.** Before this fix nothing accounted for it.
   - `logs/gpu_ledger.tsv` records the start and end of every GPU arm across lanes, and an EXIT trap costs failed or killed arms. A lane refuses to start a GPU arm once the recorded lane wall-clock hours reach `CAP_GPU_H=6`; a running arm is never stopped.
   - E1's two 4-epoch runs start only if both fit (used + 2 × (L23 + L24 lane-hours) ≤ cap). One run without its pair is useless to compare.
   - A lane that dies without an end line (SIGKILL) blocks further arms until the lead adds the line by hand.
   - The GPU-5 probe (0.16 h) was entered in the ledger by hand, because it ran under the old script.
3. **A passed probe is final.** Re-running `probe` (e.g. when relaunching a lane) now skips instead of re-deciding I3 between arms.
4. The attribution text in compare now follows §4.6 (L−R ≤ 1 SE means capacity, not pretraining; information only).
5. The script header now shows the with-sink token and step numbers. T8 token counts now include the sink.
6. Wording: the sink row was screened on val *inputs* (never val labels), not "no val".

**Real GPU probe** (run by the lead on GPU N at 03:23 UTC, outside this review):
- 27,618 padded tok/s, so the gate passes with only a 10% margin. Peak memory 20.1 GB (gate ≤ 30). One holdout eval takes 67 s.
- I3: dNDCG −0.000165 ± 0.000108, so **`--final_fp32` on every arm**. fp32 inference runs at ≈ 47K tok/s.
- At this rate: an L arm is ≈ 70 lane-min (50 train + 9 evals + 8 fp32 predictions + loading), R ≈ 85-90 lane-min, and a B arm ≈ 10-15 lane-min (measured on GPU N). Base Stage A ≈ 4.5 lane-hours, while **the E1 pair needs ≈ 4.3-4.7 lane-hours, so with the ledger above it can NOT start under CAP_GPU_H = 6.** compare then refuses to judge, as pre-declared.
- **The lead must choose before L23/L24 start:** raise `CAP_GPU_H` (≈ 10 lane-hours covers E1 on a lane this busy); run the L arms on a quieter lane (the exclusive estimate is ≈ 16 min per arm); or accept that an E1 "extend" decision leaves Stage A without a verdict.

**Open issues (not fixed; the lead's call):**
- The lead widened the GPU guards to 0-7 at 03:40 (user authorisation logged in CURRENT_STATUS). But `train_llm.py` still asserts `CUDA_VISIBLE_DEVICES ∈ {<ids>}`, so probe/L/R/E1/L2/P4B arms launched on some GPUs stop at start. B arms are unaffected.
- **P4B and L2 cannot be judged by `compare.py` as wired.**
  - A single-seed P4B gives `len(L) ≠ len(B)`, so A2(ii) cannot pass.
  - The E1 decision file asserts the L names.
  - Pre-declare how a one-seed probe is judged before running it.
- Not enforced: the 4-processes-per-GPU rule (only free memory is checked), and probe-lane vs arm-lane identity (the gate was measured on GPU N).
- The memory gate uses `max_memory_allocated` (not reserved). The I3 decision was taken at step 40.
- Qwen3.5-9B is downloading to `<mount>`, not the default HF cache, so `--backbone` needs the snapshot path (or `HF_HUB_CACHE`). Qwen3.5 on GPU still needs `flash-linear-attention` in `.venv_llm`. Stage A's 6 GPU-h cap and cost table do not cover 4B/9B arms.

---

## 1. Environment `.venv_llm` (step 1)

| item | value |
|---|---|
| interpreter | `/usr/bin/python3.12` (3.12.3), the same as `.venv` |
| creation | `uv venv --python /usr/bin/python3.12 .venv_llm` |
| torch reuse | `site-packages/zzz_project_venv_fallback.pth` = `<repo>/.venv/lib/python3.12/site-packages`. `sys.path` order: `.venv_llm` site-packages first, then `.venv`'s. `torch.__version__` = 2.11.0+cu130, imported from `.venv` |
| installed into `.venv_llm` only | `uv pip install -c <.venv freeze minus torch> transformers psutil`, then `uv pip install --no-deps peft accelerate`. `--no-deps` stops uv from pulling in a second torch, since uv cannot see `.venv`'s torch. Shared dependencies are pinned to `.venv`'s versions (numpy 2.5.4, pyyaml, tqdm, packaging, ...) |
| versions | transformers 5.22.2, peft 0.21.2, accelerate 1.15.0, tokenizers 0.23.3, safetensors 0.8.0, huggingface-hub 2.4.0. Size 271 MB |
| bytecode | `00_pycache_prefix.pth` sets `sys.pycache_prefix = .venv_llm/pycache`. No `.venv_llm` process writes `.pyc` next to sources in `.venv` (or in `nn/`) |
| `.venv` verification | `uv pip freeze` before vs after: **identical** (94 lines). Files in `.venv` newer than the pre-build snapshot: **0** |

**Incident (disclosed).**
- What happened: the first import test of the new env ran before the pycache redirect existed. Python wrote **114 bytecode-cache files** (`__pycache__/*.pyc` for PIL, jinja2, markupsafe and some torch.distributed submodules) into `.venv`.
- Cleanup: I removed exactly those 114 files and the 17 `__pycache__` directories that had been created empty. I then added the redirect `.pth`. A re-test wrote 0 files.
- Residual effect: 18 directories in `.venv` have a newer mtime. No file content in `.venv` differs from before.
- Risk to live jobs: none. `.pyc` files are caches that Python regenerates or ignores.
- Lists: `…/scratchpad/llm/build/venv_new_files.txt`, `venv_touched_dirs.txt`, `venv_freeze_{before,after,final}.txt`.

---

## 2. Code map (`<repo>/nn_x/llm/`)

| file | role |
|---|---|
| `data_llm.py` | **Read-only data layer.** `FeedStore` holds any feed subset of a split, with a compact item-embedding table. `gather()`, `batches()` and the `attach_xt` transform are copied from `nn/data.py`. `token_batches()` buckets feeds by sequence length for the LLM. `matched_subset()` is exactly `nn/train.py --train_feeds` (148,600 / 148,557 feeds for seeds 23 / 24, verified). `xt_config()` checks the copied `cvnd_pre2s` preset against `nn/train.py` (parsed with `ast`, no import). It also has a banned-input guard (`pre_*`, `*_id_contrib`, `x_lh_*`). It never creates feature links: `nn/data.feat_path` would hard-link missing files into `nn/feat_cache`. `nn/model.py` is imported read-only (`dense_features`, `listwise_loss`); `nn/common.py` is not imported, because of its GPU guard. |
| `model_llm.py` | **`LLMRanker`.** Qwen3 trunk via `inputs_embeds`, projectors, the L1/L2 masks (`build_layout`), LoRA (peft `inject_adapter_in_model`) and the heads. |
| `qwen35_branch.py` | **`Qwen35Ranker`.** Qwen3.5 trunk with the exact branch DeltaNet step (§3.3). |
| `train_llm.py` | **Trainer.** Matched subset + holdout, per-feed listwise loss + K9 aux, AdamW (two lr groups), warmup + cosine, token-budget batches, holdout evaluation every 200 steps, holdout-best checkpoint, prediction writer. |
| `run_nn_baseline.py` | **Ranker arm B.** Runs the **unmodified** `nn/train.py` with `CKPT/LOGS/PREDS` redirected to `nn_x/llm/baseline/`. It patches `nn/common`'s paths before `train` binds them, and asserts the redirect. It then writes holdout predictions in the `preds/holdout` layout so that holdout deltas can be paired. |
| `compare.py` | **Pre-declared A1/A2 evaluation**, plus the control attribution and the decision rule. |
| `estimate.py` | Exact token counts and steps for the real subsets; FLOP, time and memory estimates. |
| `run_matched.sh` | **Launch script, NOT executed.** Arms `probe B23 B24 L23 L24 R23 compare` (later `L2_*`, `P4B`). The acceptance criteria are copied into its header. It refuses GPUs other than 4-7 and refuses to start without enough free memory. |
| `smoke_test.py`, `smoke_qwen35.py`, `smoke_baseline.py` | CPU smoke tests (§4). |

Outputs of a real run, all under `nn_x/llm/`:
- LLM arms: `preds/<v>_trainonly_{val,test}.npy` (flat candidate order of `data/flat/<split>`), `preds/holdout/<v>_holdout.npy`, `ckpt/<v>.pt` (trainable tensors only), `logs/<v>.{log,json,out}`.
- Ranker arms: `baseline/{ckpt,logs,preds}`.
- Comparison: `results/compare_stageA.{json,txt}`.

---

## 3. The model as built

### 3.1 Tokens (all soft, through `inputs_embeds`; no vocabulary token is used, and the 155M-row embedding table is dropped)

> Amendment S1: one frozen vocabulary row (`<|im_start|>`) is now prepended as a sink token at position 0. See "Amendments".
- **User:** `Linear(z-normed user_emb ‖ is_user0 ‖ domain mixture) → LN → Linear(d_model) → ScaledLN`.
- **History:** `Linear(L2-normed item emb) + E_cat + E_event_type + E_event_source + E_recency + missing vector → LN → Linear(d_model) → ScaledLN`. The order is reversed to old→new, so the newest event sits next to the candidates.
- **Candidate:** the ranker's own construction (`nn/model.Ranker.forward`) at width 256, then `Linear(d_model) → ScaledLN`:
  - `Linear([e, e⊙u]) + E_cat + MLP(X) + missing → LN + domain vector`;
  - X = the ranker's `dense_features` with the ranker's normalisation: 61 base + 2 user-0 + 34 `cvnd_pre2s` inputs + 2×5 within-feed rank/z, minus 3 rel-only columns = **104**.
- **ScaledLN:** LayerNorm with its gain initialised to the RMS of the checkpoint's token embeddings. Qwen3-0.6B **0.02920**, Qwen3.5-2B 0.01501, random-init 0.0200. Every soft token therefore enters the trunk at the scale of a real token.

### 3.2 Layout L1 (default) and L2
- Token order: `[user, h_old … h_new, c_1 … c_M, pad]`.
- Mask: user and history are causal. Each candidate attends to the user, all history, and itself. Pad tokens attend only to themselves.
- Position ids: user 0, history 1..H, **every candidate H+1**.
- The mask is passed as `attention_mask={"full_attention": bool[B,1,T,T]}`. HF Qwen3 uses a prepared mask dict as-is (transformers 5.22.2, `modeling_qwen3.py:384`).
- L2 adds candidate↔candidate attention. It is built and tested, but runs only if L1 passes.

### 3.3 Qwen3.5 branch trunk (`qwen35_branch.py`)
- **Prefix:** left-padded to P = 1 + max H. All trunk maps are bias-free and map 0 to 0, so the pads keep the recurrent state and the conv window exactly as at a true sequence start.
- **Each DeltaNet layer:**
  - Runs the module's own chunked gated delta rule on the prefix, keeping its final state S_H.
  - Each candidate then takes one step from S_H with conv window `[x_{P-3}, x_{P-2}, x_{P-1}, x_c]`:
    `o_c = a_c (S_Hᵀq_c − b_c (S_Hᵀk_c)(k_c·q_c)) + b_c v_c (k_c·q_c)`
  - This is followed by the gated RMSNorm and `out_proj`.
- **Every 4th layer (gated softmax attention):** uses the L1 mask with text mRoPE.
- **LoRA targets:** additionally `in_proj_qkv`, `in_proj_z` and `out_proj`, giving 150 modules on Qwen3.5-2B.

### 3.4 Heads, trainable parameters, loss and optimiser
- **Heads:** `score = MLP(LN(h_c)) + wide(X)`; `K9 aux = MLP'(LN(h_c))`. Both run in fp32 with autocast disabled, to avoid bf16 ties under the pessimistic-tie metric.
- **Trainable parameters:**
  - LoRA r=32, α=32, dropout 0.05, B=0 at init (verified), on q/k/v/o/gate/up/down in all 28 layers: 196 modules, 20.19M parameters.
  - All RMSNorm gains (fp32 master copies), the projectors and the heads.
  - Total trainable 21.99M; the frozen body is 440.47M. Frozen weights are bf16 on GPU, fp32 on CPU.
- **Random-init control:** the same architecture from the config with HF init, all 442.2M parameters trained.
- **Loss** (the ranker's F7 terms):
  - `listwise_loss(norm="feed")` with a constant denominator token_budget/264.5 ≈ 212, so every feed weighs the same, as with `--loss_norm feed`;
  - plus `0.5·Σ(1−p_random)·CE(softmax(−pos/10), aux)/Σ(1−p_random)`.
- **Domain:** train feeds use `[1−p, p, 0]`; evaluation uses "random-like", as the ranker does.
- **AdamW** (β 0.9/0.98):
  - LoRA and norms at lr 2e-4, no weight decay;
  - projectors and heads at lr 1e-3, weight decay 0.02 on matrices;
  - control trunk at lr 5e-4, weight decay 0.02.
  - Warmup 100 steps (control 200), cosine to 10%, gradient clip 1.0.
  - Batches of 56K padded tokens (the middle of 48-64K), bucketed by length rounded up to 32. 2 epochs = **1,506 steps** for seed 23.
  - Holdout every 200 steps (8 evaluations); the holdout-best trainable state is kept. Val NDCG is logged at those steps but never selects anything.

### 3.5 Deviations from the design, and choices the design left open
- **Data on the GPU:** the L arms use `FeedStore` holding only the subset + holdout (~178K feeds, about 5 GB), instead of `SplitData` for the whole train split (35-45 GB). Gathers and dense features are bit-identical (T2).
- **Domain vector:** it sits in the user token (as specified) **and** is added to the candidate token, as the ranker does. The ranker's `wide` term is kept.
- **K9 weight:** constant 0.5. The design uses cosine with no WSD cooldown, so the F7 `--aux_anneal 0.1` has no analogue here.
- **Evaluation-count asymmetry (it comes from the design itself).**
  - The settings are L: eval_every 200 over about 1.5K steps, which is **8 evaluations**; B: eval_every 250 over about 8K steps, which is **32 evaluations**.
  - The research text expected 20-35 for both arms.
  - B gets slightly more holdout-selection optimism. That only favours B, so it makes A1 conservative. Kept as pre-declared.
- **Qwen3.5:** implemented and verified, but used only through the branch trunk, and only if escalation is ever triggered.

---

## 4. CPU smoke results (Qwen3-0.6B unless stated; 12 threads; machine load average about 335 on a shared machine)

| test | result | numbers |
|---|---|---|
| T1 preset parity + subsets | PASS | `cvnd_pre2s` identical to `nn/train.py`; n_dense 104; subsets 148,600 (s23, 6,507 with p_random>0.5) and 148,557 (s24) |
| T2 `FeedStore` == `nn/data.SplitData` (val, 80 feeds; SplitData run on CPU with `torch.cuda.synchronize` patched in-process) | PASS | All real tensors and the 104 dense features are bit-identical. The only differences are never-read filler values in masked history slots of H=0 feeds |
| T3 build | PASS | LoRA B = 0; 196 LoRA modules; frozen/trainable split correct; scores fp32 |
| T4 permutation equivariance (8 feeds) | PASS | max \|Δscore\| L1 8.4e-6, L2 8.4e-6 |
| T5 batched == single | PASS | candidate alone `[u, H, c]` vs in the batched pass 1.1e-6; feed alone vs inside a padded 4-feed batch 4.5e-6 |
| T6 no leakage | PASS | The model reads 20 input keys, never `y` or `pos`. Flipped y + reversed pos: Δ = 0.0. Feed B perturbed: feed A Δ = 0.0. L1: changing candidate 1's token moves the other candidates' hidden states by exactly 0.0 (L2: 58.4, as expected). Note: X is feed-relative by design (within-feed z/rank, as in the ranker) |
| T7 optimiser | PASS | One step on 16 train feeds: eval loss 3.9501 → 3.8912. Frozen base unchanged, LoRA B moved. First-step gradient norm 50 (clipped to 1.0). 8-feed overfit, 6 steps: 4.468 → 3.866 (not monotone: 4.468, 4.461, 4.126, 4.342, 4.086, 3.866) |
| T8 CPU throughput (8 train feeds, 4,248 padded tokens) | info | Forward 4.75 s = **0.79 TFLOP/s** on a 2N·T basis. Forward + backward 20.1 s (0.37 TFLOP/s on 4N); with checkpointing 30.6 s (**0.37 TFLOP/s on 6N**). Time ratio fwd+bwd(ckpt) / fwd = 6.5, against a FLOP ratio of 3: CPU backward is about 2x less efficient. The research's CPU estimate of 0.2-0.5 TFLOP/s holds. **This does not validate the GPU MFU; the `probe` arm exists for that** |
| T9 random-init control | PASS | 442.2M trainable; one step: eval loss 3.962 → 3.859 |
| Q1 Qwen3.5-2B naive append (L1 mask on the attention layers only) | PASS (as expected) | Scores change by up to **0.437** under a candidate permutation, so a custom mask alone cannot make Qwen3.5 order-free |
| Q2 Qwen3.5-2B branch, equivariance | PASS | max error 4.9e-5 |
| Q3 Qwen3.5-2B branch == HF forward | PASS | Maximum relative error **8.9e-6**. Reference: `Qwen3_5TextModel.forward` over the unpadded `[user, history, c]`; history lengths 241/195/207 in one left-padded batch |
| Q4 Qwen3.5-2B optimiser step | PASS | eval loss 4.290 → 4.185; 150 LoRA modules, 34.4M trainable |
| Baseline wrapper (`smoke_baseline.py`) | PASS | Redirect binds; the unmodified `nn/train.py` parser accepts the exact B flags; holdout writer layout equals `preds/holdout`; nothing written under `nn/` or `preds/` |
| `compare.py` | PASS | On stand-in F7-vs-F6 predictions it reproduces `nn/pcmp.py` exactly: +0.00167 ± 0.00077, identical slices |
| `train_llm.py` end to end (smoke mode; Qwen3-0.6B and Qwen3.5-2B, gradient checkpointing on) | PASS | Data → train → holdout/val eval → holdout-best reload → prediction writer. Metrics are meaningless at 3 steps |

**Qwen3.5-4B (local weights; same `smoke_qwen35.py`, `--backbone qwen3.5-4b`):**
- Loads as a text model: body 3,570M; 200 LoRA modules, 60.9M LoRA parameters; embedding RMS 0.01303.
- Q1, naive append: scores change by up to **0.399** under a permutation.
- Q2, branch equivariance: 1.3e-6.
- Q3, branch == HF forward: maximum relative error **4.0e-6**.
- Q4, one optimiser step on 6 feeds: eval loss 4.286 → 4.190, 84 s, peak RSS about 84 GB (fp32 without checkpointing; under the 120 GB cap).
- All PASS.

The branch trunk is therefore exact for both local Qwen3.5 sizes. The 9B uses the same code, but its weights are not local.

Result files under `/tmp/scratchpad/llm/`: `smoke/smoke_results.json`, `smoke35/smoke_qwen35_results.json`, `smoke35_4b/stdout.txt`, `e2e/logs/`, `build/` (env proofs).

---

## 5. Compute and memory (`estimate.py`; exact token counts; MFU assumed)

**Token counts** (measured from `feeds.parquet` with the trainer's own batching):

| set | feeds | mean T | real tokens | padded tokens | padding eff. |
|---|---|---|---|---|---|
| train subset s23 (per epoch) | 148,600 | 263.8 (p50 271, p90 414, max 555) | 39.21M | 41.54M (753 steps at 56K) | 0.944 |
| holdout | 30,000 | 331.3 | 9.94M | 10.40M | 0.956 |
| val | 14,423 | 270.7 | 3.90M | 4.13M | 0.946 |
| test | 28,595 | 276.5 | 7.91M | 8.35M | 0.947 |
| full train (1 epoch) | 3,151,208 | 264.5 | 833.5M | 882.8M | 0.944 |

The research assumed a padding efficiency of 0.85; the measured value is 0.944.

**Time per arm.**
- FLOP model: 6N·T for training with LoRA + checkpointing, 8N·T for the control, 2N·T for inference, +5%.
- Hardware: GPU dense BF16 peak the vendor's peak rate. Assumed MFU 15% for 0.6B and 25% for 2B/4B. **No GPU measurement exists.**

| arm | training | evaluations | final predictions | total, exclusive | pessimistic (½ MFU) | peak memory (est.) |
|---|---|---|---|---|---|---|
| L (0.6B, 2 epochs, 230.5 PFLOP) | 11.4 min | 8 × (holdout + val) 5.3 min | 1.0 min | **≈18 min/seed** | ≈36 min | ≈15 GB |
| R control (8N·T) | 15.2 min | 5.3 min | 1.0 min | ≈21 min | ≈42 min | ≈20 GB (fp32 weights + AdamW) |
| B ranker (8K steps + 32 evals + full SplitData load) | — | — | — | ≈20-30 min/seed (from project run times) | — | ≈40-45 GB |
| probe (40 steps + 1 holdout eval) | — | — | — | ≈3-5 min | — | ≈15 GB |
| **Stage A total** | | | | **≈1.8 GPU-h** | ≈3.5 h | |
| optional P4B (Qwen3-4B-Instruct-2507, 2 epochs) | 56 min | 26 min with val logging (use `--no_val_log` to cut to about 19) | 5 min | ≈1.5 h | ≈3 h | ≈40 GB |
| one full-data 0.6B epoch (Stage C) | 2.0 h | | | | 4.0 h | |

**How the memory figure for the L arm is built** (0.6B, 56K tokens):
- The residual stream is fp32, because the trainable fp32 RMSNorm gains promote it.
- Checkpointed layer inputs: 28 × 56K × 1024 × 4 B = 6.4 GB.
- Recompute working set of one layer: about 2 GB.
- Weights + LoRA + optimiser state: about 1.5 GB.
- Data: about 5 GB.

**Qwen3.5 on GPU is not estimated.** `.venv_llm` has neither `flash-linear-attention` nor `causal_conv1d`, so transformers falls back to the reference PyTorch chunked delta rule and convolution (CPU log: "falling back … much slower").
- Before any Qwen3.5 GPU run, install `flash-linear-attention` into `.venv_llm`.
- Then check that its chunk kernel back-propagates through `output_final_state`.

---

## 6. Launch protocol (`run_matched.sh`, NOT executed)

> Superseded by "Amendments → Updated protocol and cost": run the probe alone first (it makes the I3 decision), add arm `E1` after L23/L24, and val is no longer logged by the L/R arms (V0).

```
cd <repo>/nn_x/llm && mkdir -p logs
setsid nohup bash run_matched.sh <gpuA> probe L23 L24 R23 > logs/lane_gpuA.out 2>&1 & echo "$! laneA" >> logs/RUNNING.txt
setsid nohup bash run_matched.sh <gpuB> B23 B24          > logs/lane_gpuB.out 2>&1 & echo "$! laneB" >> logs/RUNNING.txt
# when all five arms are done (CPU only):
bash run_matched.sh <any 4-7> compare
```

- `<gpu>` must be 4-7, and each arm refuses to start without free memory: 30 GB for L, 35 GB for R, 60 GB for B or P4B.
- Arms are idempotent: an arm is skipped if its outputs exist.
- Read the probe's `train tok/s … padded` and `peak mem` lines before trusting §5. Padded tokens/s × 6N gives the achieved FLOP/s.
- B arm: `.venv/bin/python run_nn_baseline.py --variant llmcmp_base_s<seed> --seed <seed>` followed by the F7 stage-1 flags: `--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate --lr 2e-3 --cooldown_mix 0.1 --train_feeds 150000 --epochs 20 --eval_every 250`.
- L arm: `.venv_llm/bin/python train_llm.py --variant llm06_L1_s<seed> --seed <seed> --backbone qwen3-0.6b --layout L1 --epochs 2 --token_budget 56000 --lr_lora 2e-4 --lr_proj 1e-3 --aux_pos 0.5 --aux_tau 10 --eval_every 200 --device cuda --dtype bf16 --grad_ckpt`. R adds `--random_init --lr_full 5e-4`.

**Pre-declared acceptance** (copied into the script header; computed by `compare.py`):
- **A1 (beats the ranker):**
  - pooled paired train-only val L−B > 2 SE;
  - both seeds positive;
  - holdout ≥ 0;
  - no slice below −2 SE. Slices: val M<50, M≥50, Tue, Thu, all-cold-positive; holdout late 20%.
- **A2 (useful ensemble member):**
  - (i) L ≥ B − 0.002 on val **and** on the holdout;
  - (ii) the mean over seeds of eq(L_s, B_s) − eq(B23, B24) > 0 on val with z ≥ 1.5, and holdout ≥ 0.
  - Also reported: ρ(L, B), ρ(B23, B24), and the law-predicted gain 0.0186(1−ρ) − 0.378|gap|.

**Decision rules:**
- **Both fail:** STOP. One exception: if L ≥ B − 1 SE **and** L beats the random-init control R by more than 2 SE, run **one** Qwen3-4B-Instruct-2507 probe (`P4B`).
- **A1 passes:** run a 600K-feed slope check first; continue only if the advantage keeps at least 50%.
- **A2 passes, or the slope check passes:** full-data 0.6B members (1 epoch), then gates (a)+(b), then the frozen pool protocol.
- **L2:** only if L1 passes. **Qwen3.5:** only through `qwen35_branch.py`.
- **Divergence:** at most one pre-declared fallback (`--lora_r 64 --lr_lora 1e-4`); no search on val.

---

## 7. Known gaps and risks (none of these block the decision test)
1. **The GPU code path has not run.** Untested on CUDA: bf16 autocast with bf16 frozen weights, fp32 LoRA/norms, SDPA with a boolean 4D mask (T is rounded to a multiple of 16 for kernel alignment), fused AdamW, and non-reentrant checkpointing. The `probe` arm finds such problems in about 5 minutes.
2. **The MFU figures are assumptions.** CPU only shows that the per-token FLOP model is in the right range: forward 0.79 TFLOP/s on 12 cores. GPU throughput on a shared lane may be 2-3x below the exclusive estimates.
3. **The B arm loads the whole train split** through the unmodified `SplitData` (40-45 GB), as the design requires.
4. **The comparison is recipe-unmatched on purpose.** B uses the ranker's best recipe (Muon + WSD, 20 epochs); L uses the design's LoRA/AdamW/cosine recipe for 2 epochs. This tests "LLM as built vs our ranker as built" at matched data, not matched optimisers.
5. **Qwen3.5-9B, Qwen3-8B and Qwen3-Embedding-4B weights are not local** (index or config files only). Qwen3.5-2B and Qwen3.5-4B are local and supported. An embedding-model route has nothing to embed, because the data has no text (research §1).

## 8. Evidence vs speculation
- **Measured on CPU this session:** every PASS in §4, the token counts in §5, CPU TFLOP/s, parameter counts, and the `.venv` freeze identity.
- **Assumed:** GPU MFU (15% / 25%), the GPU peak, the B-arm run time (taken from earlier project runs, not re-measured), and the memory estimates (analytic).
- **Speculation (from the research report, unchanged):** P(A1) ≈ 10-20%, P(A2) ≈ 25-35%.
