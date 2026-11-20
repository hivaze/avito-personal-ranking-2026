# R5: full-val refit of the fine-tune for test (review_ideas #8). Nested simulation: FAIL, keep the 5-fold mean

Status: DONE 2026-11-12 00:36 UTC. All 3 pre-declared members are complete.

## TL;DR

- **Verdict: FAIL under the pre-declared rule. Keep test = mean of the 5 fold models** (`preds/nn_<v>_n2cpv3_test.npy`).
  - No `preds/nn_<v>_n2cpv3full_test.npy` was produced, and none should be used.
  - No file in `preds/`, `subs/`, `nn/`, `src/`, `data/feats/` or `pretrain/` was created or modified.

| vs (i) 5 inner fold models (pooled per feed, 3 members, n = 14,423) | delta | z | members (v15cd / v14 / v17) | rule |
|---|---|---|---|---|
| (ii) one refit on the whole fit set | **−0.00025 ± 0.00018** | −1.37 | −0.00028 / −0.00054 / +0.00007 | **FAIL** (not > 2 SE; 2 members < 0) |
| (iii) 0.5 (i) + 0.5 (ii), raw scores | **−0.00002 ± 0.00012** | −0.18 | +0.00019 / −0.00018 / −0.00007 | **FAIL** (not > 2 SE; 2 members < 0) |

- **Mechanism: averaging 5 fine-tunes is worth more than +25% data for one model.**
  - A second, independent seed comes for free. The production OOF model for fold k is a fine-tune on exactly the same 80% (same GroupKFold call, same recipe).
  - That model is also below (i): **−0.00052 ± 0.00017 (z −3.1)**, negative on all 3 members and all 6 slices.
  - So two independent seeds of a single 80% fine-tune both lose to the 5 × 64% fold mean.
  - Their A/A difference (+0.00027 ± 0.00019) is seed noise.
- **Post hoc (not pre-declared):** a 2-seed refit (0.5 (ii) + 0.5 production fold model) is still **−0.00022 ± 0.00016** vs (i).
  - A "refit with k seeds" variant would need about 5 seeds on 100% of val per member just to match today's 5 folds, at the same GPU cost, with an expected gain of about 0.
  - Idea #8 is closed.
- **What to submit right now:** nothing from this experiment (§5).
  - Cd (v13) is already on public at 0.39208.
  - The only gated file never checked on public is pick 1, `subs/sub_v12_nnfinal5v3mix.parquet`. It is worth one submission as a construction sanity check only, not for picking between v12 and v13.

## 1. Pre-registration (written before any refit run)

`/tmp/scratchpad/round5/refit/PREREG.txt` was written before the first GPU call (23:07 UTC):

- **Members:** v15_cd_all_e10 (F5cd), v14_s93 (F6), v17_pre2s_s102 (F7).
- **Outer split:** GroupKFold(5) by user_id, with every user-0 feed as a singleton group.
  - This is the same call as `nn/finetune.py --u0_singleton`, and sklearn GroupKFold is deterministic.
  - So the outer folds are exactly the production fine-tune folds.
- **Per outer fold k** (R_k = the other 80%, about 11,538 feeds):
  - (i) inner GroupKFold(5) on R_k (same group rule), 5 fine-tunes on about 64% each; the raw-score mean of their final-EMA predictions on fold k;
  - (ii) one fine-tune on all of R_k;
  - (iii) 0.5 (i) + 0.5 (ii) on raw scores (what a production blend of fold-mean and refit test files would do).
- **Seeds:** inner j of outer k = 1000j + 10k + 1; refit of outer k = 9000 + 10k + 1.
- **Recipe:** identical to the live n2cpv3 fine-tune (`--extra n2c_pre_v3 --lr 5e-5 --wd 0.02 --warmup 30 --decay cos --ema 0.99 --ema_start 0.5 --epochs 4 --u0_singleton`, cand_budget 8192, max_feeds 128), final EMA weights, no early stopping.
  - Same epochs means 25% more steps for the refit: 696-712 vs 568-584 steps per fine-tune (logs).
- **Acceptance:** x in {ii, iii} passes iff the pooled per-feed delta(x − i) (member deltas averaged per feed, SE over the 14,423 feeds) is > 2 SE AND every member's own mean delta is >= 0.
  - If both pass: prefer (iii) unless (ii) − (iii) > 2 SE.
  - Slices are reported only.
- **Declared caveat:** the simulation compares 64% vs 80% of val; production compares 80% vs 100%. The data ratio is the same (×1.25), but with diminishing returns the production benefit of the refit should be, if anything, smaller.

## 2. Code, hygiene, parity check

- `nn_x/solvers/refit_ft.py` (new). GPU N only, enforced by `nn_x/solvers/common.py:13`.
  - **Training loop:** copied line by line from `nn/finetune.py`: warm-up, cosine, AdamW groups with extra_lr_mult 10, clip 1.0, EMA start/lerp, final EMA prediction.
  - **Dropped:** the unused pseudo/add_random branches and the per-epoch predictions. `predict()` draws no random numbers, so this does not change training.
  - **Preset:** `n2c_pre_v3` is rebuilt by the same chain as `nn/finetune.py` (42 features); the script asserts that none is missing.
  - **I/O:** checkpoints are read-only from `nn/ckpt`. Outputs go only to `nn_x/solvers/preds/refit/` and `nn_x/solvers/logs/r5_refit_*`.
  - `nn_x/solvers/model.py` is byte-identical to `nn/model.py` (`cmp`), and `SplitData.batches`/`gather` are identical to `nn/data.py`.
- **Parity check** (`--mode check`: original fold-0 split, original seed 0, v15_cd_all_e10) vs the stored `preds/nn_v15_cd_all_e10_n2cpv3_val_oof.npy` on fold 0:
  - corr 1.000000, max abs diff 0.042 logits, NDCG 0.40047 vs 0.40065 (GPU nondeterminism), 712 steps (`nn_x/solvers/logs/r5_refit_check.log`).
  - The copied recipe reproduces production.
- **Evaluation:** `CUDA_VISIBLE_DEVICES="" python solvers/r5_refit_eval.py nested` (CPU, 16 threads) -> `solvers/out/r5_refit/nested.json`.
  - It uses `src/metrics.ndcg_per_feed` (pessimistic ties).
  - Every val feed is scored exactly once, by the model set of its outer fold, which never saw it.
  - Day split as in `nn/pcmp.py`. Cold tercile = top third of feeds by share of candidates with it_hist_all_luo == 0 (n = 4,819).

## 3. Results

### 3.1 Per member (NDCG over all 14,423 val feeds; deltas ± per-feed SE)

| member | (i) 5 × 64% mean | (ii) 1 × 80% refit | (iii) mix | prod OOF (1 × 80%, 2nd seed) | (ii) − (i) | (iii) − (i) | prod − (i) | A/A (ii) − prod |
|---|---|---|---|---|---|---|---|---|
| v15_cd_all_e10 | 0.39536 | 0.39509 | 0.39555 | 0.39497 | −0.00028 ± 0.00029 | +0.00019 ± 0.00020 | −0.00039 ± 0.00027 | +0.00011 ± 0.00031 |
| v14_s93 | 0.39616 | 0.39562 | 0.39598 | 0.39570 | −0.00054 ± 0.00033 | −0.00018 ± 0.00023 | −0.00046 ± 0.00032 | −0.00008 ± 0.00035 |
| v17_pre2s_s102 | 0.39630 | 0.39637 | 0.39623 | 0.39559 | +0.00007 ± 0.00028 | −0.00007 ± 0.00019 | −0.00071 ± 0.00027 | +0.00078 ± 0.00029 |

- Within-feed z correlation between (i) and (ii): 0.9994 / 0.9991 / 0.9994.
- The v17 A/A of +0.00078 (z 2.7) shows how large single-seed fine-tune noise is. This matches CURRENT_STATUS §3k: the fine-tune A/A spread is 1.4× the per-feed SE.
- That is why the member rule and the second seed (prod) matter.

### 3.2 Pooled over the 3 members (per-feed mean of member deltas)

| comparison | all (14,423) | M<50 (5,893) | M>=50 (8,530) | Tue (7,543) | Thu (6,880) | cold tercile (4,819) |
|---|---|---|---|---|---|---|
| **(ii) − (i)** (decision) | **−0.00025 ± 0.00018** | −0.00022 ± 0.00033 | −0.00027 ± 0.00021 | −0.00022 ± 0.00025 | −0.00029 ± 0.00026 | −0.00013 ± 0.00034 |
| **(iii) − (i)** (decision) | **−0.00002 ± 0.00012** | −0.00024 ± 0.00023 | +0.00013 ± 0.00013 | −0.00011 ± 0.00017 | +0.00008 ± 0.00017 | −0.00014 ± 0.00022 |
| (ii) − (iii) | −0.00023 ± 0.00013 | +0.00002 ± 0.00022 | −0.00040 ± 0.00016 | −0.00010 ± 0.00017 | −0.00036 ± 0.00019 | +0.00001 ± 0.00023 |
| (iii, z-mean) − (i) (info) | −0.00003 ± 0.00012 | −0.00027 ± 0.00023 | +0.00014 ± 0.00013 | −0.00013 ± 0.00017 | +0.00009 ± 0.00017 | −0.00017 ± 0.00022 |
| prod − (i) (independent 80% seed) | −0.00052 ± 0.00017 | −0.00065 ± 0.00032 | −0.00043 ± 0.00018 | −0.00050 ± 0.00023 | −0.00054 ± 0.00025 | −0.00057 ± 0.00032 |
| A/A (ii) − prod | +0.00027 ± 0.00019 | +0.00043 ± 0.00034 | +0.00016 ± 0.00022 | +0.00029 ± 0.00027 | +0.00025 ± 0.00027 | +0.00044 ± 0.00035 |
| POST HOC 2-seed refit − (i) | −0.00022 ± 0.00016 | −0.00017 ± 0.00030 | −0.00026 ± 0.00017 | −0.00021 ± 0.00022 | −0.00024 ± 0.00023 | +0.00003 ± 0.00029 |

### 3.3 Per outer fold, delta vs (i)

| member | (ii) − (i), folds 0..4 | (iii) − (i), folds 0..4 |
|---|---|---|
| v15_cd_all_e10 | −0.00138, +0.00047, +0.00025, −0.00014, −0.00059 | +0.00052, −0.00033, +0.00064, +0.00008, +0.00003 |
| v14_s93 | −0.00112, +0.00010, −0.00016, −0.00036, −0.00117 | −0.00065, −0.00016, +0.00079, −0.00046, −0.00045 |
| v17_pre2s_s102 | −0.00026, +0.00032, +0.00095, −0.00055, −0.00012 | −0.00038, −0.00027, +0.00042, −0.00030, +0.00017 |

(ii) is below (i) in 9 of 15 member-folds.

### 3.4 Reading

- The hypothesis (+0.0002 to +0.0005 for a 100% refit) is not supported. The refit point estimate is negative.
  - The independent-seed replication is negative at z −3.1.
  - The mix (iii) is exactly neutral.
- **Why:** the fine-tune moves the train-only model only slightly (z corr 0.999). Five fold models therefore differ mainly in fine-tune noise, and averaging that noise out is worth more than the +25% labels one model gains from a refit.
  - This matches the high fine-tune A/A spread.
- **Staleness:** no slice reverses the picture. Thu and the cold tercile, the most private-like slices, are within 1 SE of 0 for (ii) and (iii). Nothing suggests the refit would help more on the colder private split.
  - Stale-val was not run. The refit is not a staleness intervention, and the fine-tune extras (it_hist_*_luo rank/z/gap derivatives) have no stale counterparts in the replay tooling.
- **Ensemble level:** Cd uses 19 members × 5 fold models. Replacing each member's fold mean with one refit would cut the fine-tuned branch from 95 to 19 models. The member-level evidence gives no reason to expect a gain.

## 4. Decision and deliverables

- **Pre-declared decision:** (ii) FAIL, (iii) FAIL -> choice = **none: keep the 5-fold mean** for every Cd member. `solvers/out/r5_refit/nested.json` -> `decision.choice`.
- Not produced (by the rule): `preds/nn_<v>_n2cpv3full_test.npy`. The full-refit stage of `nn_x/solvers/run_r5_refit.sh` never ran (§6).
- Closed: review_ideas #8, including the "refit with seeds" variant (post-hoc 2-seed refit −0.00022 ± 0.00016).
- The only cheap fine-tune-side lever this points to is more fold/seed averaging, not more data per model. For example, `finetune.py --seeds 2` doubles fine-tune cost. That is a hypothesis only and was not tested here; it would need its own pre-declared paired check.

## 5. What can be submitted right now (user question)

- **Already on public:** `subs/sub_v13_Cd_memeq_F5cdF6F7.parquet` = **0.39208** (CURRENT_STATUS 23:20; predicted 0.3909, residual +0.0012). This is pick 2.
  - I re-checked it: byte-identical to `subs/cand_Cd_memeq_F5cdF6F7.parquet` (`cmp`); 2,904,806 rows equal to sample_submission; 0 missing keys, 0 duplicate keys, 0 within-feed ties, all finite.
- **The only ready, gated, never-publicly-checked file is `subs/sub_v12_nnfinal5v3mix.parquet` (pick 1).**
  - presub_gate PASS with 9 WARN (CURRENT_STATUS §3m); val 0.39818; predicted public 0.3900 / private 0.3927.
  - Submitting it only checks that pick 1 is built correctly on test: alarm if public < ~0.387 (prediction − 0.003).
  - It must not be used to choose between v12 and v13. The val gap (+0.00083 ± 0.00054) is below the public SE (~0.0012), and the frozen protocol forbids picking on public differences < 0.003.
- **Nothing from this experiment:** it failed, and even a pass (+0.0002 to +0.0005 expected) would be invisible on public.

## 6. Process notes

- **Pause incident:** the first driver (`nn_x/solvers/run_r5_refit.sh`: nested, then full refits) was paused by me with SIGSTOP at 23:48, so that the full-refit stage would not auto-start before the decision.
  - The harness treats a stopped background task as finished (exit 147) and killed its process group. Member 3 died about 1 minute after starting.
  - Members 1-2 were already saved: `nested_v15_cd_all_e10.npz` 23:28, `nested_v14_s93.npz` 23:48.
  - No full refit ever ran.
  - Lesson: never SIGSTOP a harness background task; gate later stages inside the script.
- **GPU N process limit:** CURRENT_STATUS 23:45 sets at most 4 training processes per GPU. GPU N was at 4 (GRU s501, DIN s601, agedrop s23/s24) when I restarted member 3, so I stopped my run.
  - It was relaunched by a slot-aware wrapper (`.../round5/refit/wait_and_run_s102.sh`: launch only with <= 3 processes and >= 60 GB free on GPU N).
  - It launched at 00:20:13 (3 processes, 95 GB free) and finished at 00:35:22.
  - The seeds and design are unchanged; the seeds depend only on (k, j).
- **Ordering:** the decision was already fixed after 2 members, because both variants had a negative member. Member 3 was completed for the full pre-declared design and did not change the outcome.
- **Compute:** GPU N only, ≤ 1 process of mine at any time. CPU evaluation ran with `CUDA_VISIBLE_DEVICES=""` and 16 threads.

## Files

- Code:
  - `nn_x/solvers/refit_ft.py`
  - `nn_x/solvers/run_r5_refit.sh`
  - `solvers/r5_refit_eval.py`
- Pre-registration and launcher:
  - `/tmp/scratchpad/round5/refit/PREREG.txt`
  - `/tmp/scratchpad/round5/refit/wait_and_run_s102.sh`
- Raw outputs:
  - `nn_x/solvers/preds/refit/nested_{v15_cd_all_e10,v14_s93,v17_pre2s_s102}.npz` (P_i, P_ii, outer fold id)
  - `nn_x/solvers/preds/refit/check_v15_cd_all_e10_fold0.npy`
- Logs:
  - `nn_x/solvers/logs/r5_refit_check.log`
  - `nn_x/solvers/logs/r5_refit_nested.log`
  - `nn_x/solvers/logs/r5_refit_driver.stdout`
  - `nn_x/solvers/logs/r5_refit_nested_s102.stdout`
- Stats: `solvers/out/r5_refit/nested.json`
