# CPU3 / W6: the K9 production-order aux head as a signal. Stage 1 FAILS the frontier gate, so W6 is CLOSED

Code: `solvers/CPU3_aux_head_member_W6.py` (subcommands `replay`, `stage1`, `stage2`, `diag_val`, `fidelity`). The replay harness is in `solvers/auxexp/{data,model}.py`, and the extra check is `solvers/auxexp/mainprod_check.py`.
CPU only (`CUDA_VISIBLE_DEVICES=""`). The replay ran as 2 workers × 8 threads; the analysis used 16 threads.
Outputs:
- Replays: `solvers/out/auxexp/<member>_val_{main_rl,aux_rl,main_prod,aux_prod}.npy` (float32, aligned with the val candidate rows; 48 files).
- Analysis JSON and logs: `solvers/out/CPU3_aux_head_member_W6/` (`stage1.{json,log}`, `diag_val.{json,log}`, `fidelity_train_prod.json`, `fidelity.log`, `mainprod_vs_mainrl.{json,log}`, `replay_val_F{5,6}.log`).

Stage 2 and the test replay did **not** run, as the pre-declared stop rule requires. No test predictions were produced, and nothing was written to `preds/`, `nn/` or `subs/`.

## TL;DR

- **The gate fails by 2×. W6 is closed, and no blend was attempted.**
  - Primary signal: the 12-member z-mean of `aux_prod` (aux head under the production domain one-hot [1,0,0]).
  - Its val NDCG is **0.30355** ± 0.0016, against **0.39770** for the main train-only z-mean of the same 12 members. The gap is 0.09414.
  - The within-feed correlation with the main z-mean is **ρ = 0.0335**.
  - The break-even gap is 0.049(1 − ρ) = 0.04736. The actual gap (0.0941) is above it, so the gate FAILS.
- The secondary signal `aux_rl` (random-like domain, the default at val/test inference) is no better: NDCG 0.30249, ρ 0.0062, gap 0.0952 against a break-even of 0.0487. It also FAILS.
- The aux output is nearly uninformative for exploration clicks:
  - A random order scores 0.28876 on val. aux_prod is only +0.015 above that, while the main head is +0.109 above it.
  - aux_prod is almost orthogonal to the main head (ρ 0.03) and anti-correlated with popularity. Its within-feed correlation with log1p(it_hist_all_luo) is −0.31, against +0.28 for the main head.
- **What this means for arms A-C (P1, K9 tuning):** the K9 output carries no usable extra ranking signal.
  - The +0.0024 that K9 gave (nn_report.md:141) must come from shaping the shared trunk (multi-task regularisation), not from a production-order estimate that the main head fails to use.
  - So judge K9 variants only on the main head (train-only val + holdout). Do not export or blend the aux output.
- Replay check (required): the CPU replay of the main logit correlates **0.9999983-0.9999993** with the stored `preds/nn_<m>_trainonly_val.npy` for all 12 members. The threshold was ≥ 0.99999, so it PASSES.
- Unplanned side observation (not a pre-declared test, not actionable): running the *main* head under the production domain is a pure cold/warm trade.
  - Overall it is +0.00011 ± 0.00017 against the default domain, pooled over 12 members.
  - On all-cold-positive feeds it is +0.0026 ± 0.0004 (10/12 members positive), so the other feeds lose about 0.0007.
  - Net zero, so do not pursue (section 5).

## 1. Setup and hygiene (verified)

- `solvers/auxexp/` holds copies of `solvers/errors/nncpu/{data,model}.py`. I checked before patching: `diff` of the nncpu `model.py` against `nn/model.py` was empty, and the nncpu `data.py` differs from `nn/data.py` only by the read-only `feat_path` patch and the removed cuda syncs.
- `data.py` (same no-write patch as the CPU2 harness, `solvers/stale/`):
  - `feat_path` reads from `nn/feat_cache/<split>/` with a fallback to `data/feats/`, and never creates a hard link or copy.
  - The unused original write-path function was **deleted** from the copy.
  - `grep -n "save\|link\|copy2\|mkdir\|open("` finds nothing in `data.py` or `model.py`.
  - `find nn/feat_cache -newer solvers/auxexp/model.py` lists no files.
- `model.py` patches (`diff nn/model.py solvers/auxexp/model.py`):
  - (a) `if self.aux_pos and self.training:` → `if self.aux_pos:` (nn/model.py:498). The aux head is now computed in eval mode, in fp32.
  - (b) A new attribute `dom_override` (a [3] vector). If the batch has no `"dom"`, it replaces `one_hot(infer_domain)`.
  - For val/test the batch never has `"dom"` (`nn/predict.py` sets no `d.dom`), so the default stays random-like (`infer_domain = 1`), exactly as the stored train-only preds.
  - The domain vector enters twice, as in training: once on the candidate tokens before the cross/set blocks, and again on the head input (`nn/model.py:428-433`, `:492-493`). The aux head reads the representation *after* the second addition, the same place it read it in training.
  - Smoke test on 300 feeds: passing `dom_override=[1,0,0]` gives logits identical (max abs diff 0.0) to putting `b["dom"]=[1,0,0]` in the batch.
- Both passes reuse the same gathered batch and dense features. The main logit and aux output are taken from each pass.
- Members: the 12 train-only checkpoints, read-only from `nn/ckpt/`.
  - F5: v12_all_e10, v12_all_s81, v12_all_s82, v12_allprew_s83/84/85.
  - F6: v13_lr1_mix1, v14_s92..s96.
  - Every checkpoint has `aux_pos=True` (listwise K9, τ 10, aux_pos 0.5 annealed to 0.1), `domain=True`, `arch=tf`, `retr_m 16`, and no ID buckets.
- Replay check (`stage1.log`, `replay_val_F*.log`): the main_rl correlation with the stored train-only val preds is 0.9999991 / 0.9999991 / 0.9999993 / 0.9999992 / 0.9999990 / 0.9999993 (F5) and 0.9999985 / 0.9999984 / 0.9999984 / 0.9999989 / 0.9999985 / 0.9999983 (F6).
  - The max abs diff of 0.5-1.06 logits comes from bf16 GPU autocast vs fp32 CPU.
  - The CPU main_rl z-mean scores 0.39749 against 0.39770 for the stored files (−0.00020 ± 0.00018, noise).
  - All stage-1 comparisons use the **stored** main preds as the reference.
- Cost: about 110-138 s per member for both passes on val (2 workers × 8 threads), about 12 min for all 12 members.

## 2. Stage 1 (pre-declared gate: gap ≤ 0.049(1 − ρ); sol_ensemble.md §2.5)

- ρ is the mean within-feed Pearson correlation of the two per-feed z-scored 12-member z-means. This is the same definition as `solvers/ensemble/exp7_pairs.py:pf_corr`, the one the break-even law was fitted with.
- gap = NDCG(main TO z-mean) − NDCG(signal z-mean).

| signal (12-member z-mean) | val NDCG | member range | ρ with main TO | gap | break-even 0.049(1−ρ) | gate |
|---|---|---|---|---|---|---|
| main train-only (stored preds) | **0.39770** | | | | | (reference) |
| **aux_prod** (primary, domain [1,0,0]) | **0.30355** ± 0.0016 | 0.29038-0.31852 | **0.0335** | **0.09414** | 0.04736 | **FAIL** |
| aux_rl (secondary, domain [0,1,0]) | 0.30249 | 0.28235-0.31511 | 0.0062 | 0.09521 | 0.04870 | FAIL |
| main_prod (information only) | 0.39780 | 0.39116-0.39431 | 0.9982 | −0.00011 | 0.00009 | (not a W6 signal) |
| main_rl (replay check) | 0.39749 | 0.39119-0.39404 | 0.9999 | 0.00020 | 0.00000 | (replay) |

aux_prod standalone minus main, per slice (paired per feed, `stage1.log`):

| slice | all | M<50 | M>=50 | Tue | Thu | all-cold-positive feeds |
|---|---|---|---|---|---|---|
| aux_prod − main | −0.0941 ± 0.0020 | −0.1287 ± 0.0038 | −0.0702 ± 0.0021 | −0.0942 ± 0.0028 | −0.0941 ± 0.0029 | +0.0001 ± 0.0043 |

- **Decision: STOP.** The gap is 2.0× the break-even. ρ is so low that the gap would have to be below 0.047. It is 0.094.
- The secondary signal fails just as clearly.
- In stage 1 I computed no blend of any kind (no equal mix, no w-grid), as the stop rule requires.
- The only slice where aux_prod reaches the main head is the all-cold-positive feeds (23.7% of val; +0.0001 ± 0.0043). That fits the anti-popularity profile in §3: production ranks cold/fresh listings high.

Member-level within-feed correlations (`stage1.json: member_diag`):
- ρ(aux_prod, own main) has a mean of 0.023.
- ρ(aux_rl, own main) has a mean of −0.003.
- ρ(aux_prod, aux_rl) has a mean of 0.862. The domain token moves the aux output somewhat, but both versions are equally weak.

## 3. Why the aux output is weak (diagnostics, no blend)

**Val context** (`diag_val.log`):
- Random order scores 0.28876 (sd 0.0013 over 5 draws). aux_prod at 0.30355 is +0.015 above random. The main head is +0.109.
- The f-feature profile differs between the two heads:

| feature | within-feed corr with aux_prod z-mean | with main TO z-mean | single-feature NDCG (+f / −f) |
|---|---|---|---|
| f1 | +0.29 | +0.49 | 0.328 / 0.240 |
| f2 | +0.15 | −0.05 | 0.236 / 0.256 |
| f3 | +0.27 | +0.45 | 0.252 / 0.236 |
| f4 | **−0.26** | +0.28 | 0.294 / 0.266 |
| f5 | −0.02 | +0.27 | 0.295 / 0.255 |
| log1p(it_hist_all_luo) | **−0.31** | **+0.28** | |

  Some single-feature NDCGs are below random because f has ties and ties are scored pessimistically.

**Fidelity to the production order** (`fidelity_train_prod.json`):
- Setup: 3,000 production train feeds (p_random ≤ 0.05, 20 ≤ M ≤ 400, mean M 94.7, fixed seed 777, not in the holdout).
- These feeds are **in-sample** for the models, so this only checks that the head works in eval mode.

| score (v12_all_e10 / v14_s92) | within-feed Spearman with −pos | top-1 = production slot 0 | click NDCG on these feeds |
|---|---|---|---|
| aux_prod | 0.286 / 0.288 | 21.5% / 21.4% | 0.340 / 0.337 |
| aux_rl | 0.272 / 0.268 | 20.6% / 21.3% | 0.343 / 0.340 |
| main_rl | 0.112 / 0.115 | 13.0% / 13.5% | 0.377 / 0.381 |
| main_prod | 0.111 / 0.121 | 14.4% / 16.3% | 0.378 / 0.383 |
| −pos (the production order itself) | 1 | 100% | **0.335** |

- The head does imitate the top of the production order: top-1 agreement is about 21%, against about 1% for chance at mean M 95. The full-list Spearman is only 0.29, which is expected: the τ = 10 target puts about 63% of its mass on the first 10 slots.
- **Verified:** on production feeds, where clicks are position-biased toward the top, the production order itself has click NDCG 0.335. That is about what the aux head scores (0.337-0.343), and well below the main head (0.377-0.383).
  - Caveat: the main head is in-sample here, so its margin is optimistic.
  - Still, the logged `pos` order is a weak predictor of clicks in this data. Its imitation is weaker still on exploration feeds.
- **Interpretation (suspected, consistent with all of the above):**
  - The production ranker promotes fresh, low-popularity listings (the aux output is anti-correlated with LUO popularity and with f4) and optimises an objective other than clicks.
  - So its order adds no click information that the main head lacks.
  - K9's measured benefit (+0.0024 val / +0.0027 holdout, nn_report.md:141) is better explained as auxiliary-task regularisation of the shared trunk.
  - Consequence for P1 (K9 τ / Plackett-Luce arms): the expected gain mechanism is "a better trunk", so measure it only on the main head. The aux output itself should never enter a blend.

## 4. Stage 2

Not run. `stage2()` asserts `stage1.json: signals.aux_prod.gate_pass` and refuses to run when the gate fails. The pre-declared nested GroupKFold(5) choice of w ∈ {0.1, 0.25} was therefore not evaluated.

## 5. Unplanned side observation: the domain token is a cold/warm knob for the *main* head (information only)

The prod pass also produced main_prod for free. Paired with main_rl (both CPU fp32, same batches; `mainprod_vs_mainrl.log`):

| | all | M<50 | M>=50 | Tue | Thu | all-cold-positive (23.7% of feeds) |
|---|---|---|---|---|---|---|
| main_prod − main_rl, pooled over 12 members | +0.00011 ± 0.00017 | +0.00032 ± 0.00030 | −0.00004 ± 0.00020 | +0.00020 ± 0.00024 | +0.00001 ± 0.00024 | **+0.00262 ± 0.00036** |

- Member level: the overall delta has a mean of +0.00011 (sd 0.00034, positive in 6/12). The all-cold delta has a mean of +0.0026 (sd 0.0021, positive in 10/12).
- So the other 76% of feeds lose about 0.0007. The domain shift moves the main head toward cold items without changing the total.
- Not actionable:
  - The net is zero.
  - This was not a pre-declared test, and the slice was chosen post hoc among 6 slices × 4 signals.
  - The fine-tune half already re-calibrates cold handling (exploration-domain token, `nn/finetune.py:113`).
- It is mainly evidence that the domain embedding carries a "production favours fresh items" prior.
- If anyone wants to use it, the only principled form would be a pre-declared test on a colder population (stale-val emulation, as CPU2's harness does), not on val.

## 6. Reproduce

```
cd <repo>
AUX_THREADS=8 CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/CPU3_aux_head_member_W6.py replay val <members...>   # ~2 min/member
AUX_THREADS=16 CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/CPU3_aux_head_member_W6.py stage1
AUX_THREADS=16 CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/CPU3_aux_head_member_W6.py diag_val
AUX_THREADS=16 CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/CPU3_aux_head_member_W6.py fidelity v12_all_e10 v14_s92   # loads train (~35 GB RAM), ~6 min
CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/auxexp/mainprod_check.py
```
