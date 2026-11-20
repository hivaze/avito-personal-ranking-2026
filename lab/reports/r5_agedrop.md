# Round 5: age-conditioned ID-input dropout (review_ideas #6), 2-epoch paired A/B

Status: DONE 2026-11-12 00:35 UTC. VERDICT: FAIL (stale-val +0.00089 ± 0.00062, z 1.44 < 2; holdout −0.00189 ± 0.00034, z −5.6). Pre-declaration written 2026-11-11 23:15 UTC, before any treatment run was launched. Results are appended below
the pre-declaration and do not change it.

## 0. Pre-declaration (frozen before launch)

**Arm (one treatment, no other arms):** `sx_agedrop_s23`, `sx_agedrop_s24` = the 2-epoch harness recipe of
`nn_x/solvers/run_sx_lane.sh` (cvnd_pre2s, Muon lr 1e-3, WSD cooldown 0.2, mix 0.1, retr16 + attn gate, K9 listwise
tau 10, holdout-selected) plus `--agedrop prv_rl13`. Launched as
`bash nn_x/solvers/run_sx_lane.sh agedrop "--agedrop prv_rl13" <seed>` for seeds 23 and 24 (two lanes on GPU N).

**Control:** the existing `sx_base_s23`, `sx_base_s24` (same lane script, same train.py apart from the flag). Their
args are diffed against the treatment's after the runs; the only allowed difference is `agedrop`.

**Mechanism.** During training only, the 8 ID-based train-stage inputs of a candidate row (`seq_cv_invd20`,
`seq_cv_pop`, `seq_cv_norm`, `it_hist_all_luo`, `it_hist_et1..4_luo`; train.py's `xt_idcols` rule) are set to 0.0 with
probability p(age bucket). The in-model rank/z channels of `seq_cv_invd20` and `it_hist_all_luo` (XT_REL) are computed
after the blanking. Inference and `model.cfg` are unchanged.

**Age.** age(f4) in hours via the life_ knots (`data/feats/life_f4_age_knots.npy`; reports/feat_life_.md sec 1: f4 is
the listing's exact age at the feed time). It is row-local, label-free and computed identically on every split.
f4 >= 63/64 (the 4.6-year cap) is its own bucket. 13 buckets: 0-6h, 6-24h, 1-2d, 2-3d, 3-5d, 5-7d, 7-14d, 14-30d,
30-60d, 60-100d, 100-200d, >200d, capped.

**Probabilities (rule, then numbers).** Blank state z = all 8 ID inputs are 0 after attach_xt (raw NaN or <= 0).
For each bucket, p = clip((z_private − z_train_rl) / (1 − z_train_rl), 0, 1).
- z_train_rl: train random-like rows (p_random > 0.5, holdout feeds excluded; the same exploration candidate generation
  as val/test, so the difference isolates coverage, not production selection).
- z_private: test feeds with timestamp >= 1790497920.
- Source: `scratchpad/round5/agedrop/measure.py` -> `measure.json` (section 1).

| bucket | 0-6h | 6-24h | 1-2d | 2-3d | 3-5d | 5-7d | 7-14d | 14-30d | 30-60d | 60-100d | 100-200d | >200d | capped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p (`prv_rl13`) | 0.961 | 0.937 | 0.898 | 0.862 | 0.833 | 0.810 | 0.578 | 0.164 | 0.063 | 0.048 | 0.051 | 0.094 | 0.097 |

Expected effect on the training rows: blank share 0.128 -> 0.437 (private 0.456); 30.8% of training rows change.

**Yardsticks (all train-only, paired per feed, the 2 seed pairs averaged per feed before the SE).**
1. Stale-val (PRIMARY): CPU2 replay tooling, scenario `both_pre2both`:
   `python solvers/CPU2_stale_replay_pre2s_and_arms.py C sx_base_s23:sx_agedrop_s23,sx_base_s24:sx_agedrop_s24 phaseC_agedrop`.
2. Val: `python nn_x/solvers/sx_eval.py base agedrop` (stored train-only val preds).
3. Holdout: same script (30K random-like train feeds, M >= 50; inputs carry dense future counts).

Slices reported for every yardstick where defined: M<50 / M>=50 / Tue / Thu (val, stale-val), all-cold-positive /
all-warm-positive (stale-val), late 20% (holdout), plus per-seed deltas.

**Acceptance (pre-declared, from the task):** PASS iff
- stale-val pooled delta > 2 SE, AND
- val pooled delta >= −1 SE, AND
- holdout pooled delta >= −1 SE.

Nothing is tuned after results are seen. If it passes, the 10-epoch family spec is written in section 6 for the lead
(no 10-epoch run is launched here). If it fails, the verdict is FAIL and no variant is tried in this round.

**Validity checks (must hold, else the A/B is void):** CPU unit test passes; the treatment's ckpt cfg == the
baseline's cfg (stock model.py loads it); stale replay check (d) corr >= 0.99999 for the new checkpoints; the logged
realised blank share per bucket matches the targets.

## 1. Measurement behind the probabilities (label-free; CPU, 29 s)

`scratchpad/round5/agedrop/measure.py` (log `measure.log`, numbers `measure.json`). Blank-state share by age bucket:

| bucket | train rows (all, no holdout) | train random-like | holdout | val | val Tue | val Thu | public | private | % train rows | % private rows |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-6h | 0.409 | 0.579 | 0.595 | 0.941 | 0.936 | 0.947 | 0.962 | 0.984 | 6.85 | 6.24 |
| 6-24h | 0.217 | 0.422 | 0.455 | 0.900 | 0.893 | 0.908 | 0.927 | 0.964 | 6.71 | 4.77 |
| 1-2d | 0.167 | 0.319 | 0.346 | 0.856 | 0.848 | 0.863 | 0.886 | 0.930 | 4.98 | 4.06 |
| 2-3d | 0.148 | 0.266 | 0.288 | 0.825 | 0.799 | 0.848 | 0.868 | 0.899 | 3.60 | 3.26 |
| 3-5d | 0.136 | 0.227 | 0.244 | 0.701 | 0.602 | 0.810 | 0.842 | 0.871 | 5.17 | 5.53 |
| 5-7d | 0.115 | 0.193 | 0.209 | 0.515 | 0.438 | 0.613 | 0.818 | 0.847 | 3.80 | 4.60 |
| 7-14d | 0.100 | 0.172 | 0.187 | 0.364 | 0.328 | 0.404 | 0.513 | 0.650 | 8.33 | 9.50 |
| 14-30d | 0.087 | 0.155 | 0.170 | 0.224 | 0.212 | 0.237 | 0.261 | 0.294 | 10.12 | 11.77 |
| 30-60d | 0.070 | 0.131 | 0.146 | 0.169 | 0.164 | 0.174 | 0.178 | 0.185 | 9.25 | 10.45 |
| 60-100d | 0.076 | 0.128 | 0.141 | 0.162 | 0.159 | 0.166 | 0.167 | 0.170 | 7.35 | 7.53 |
| 100-200d | 0.090 | 0.139 | 0.150 | 0.175 | 0.173 | 0.177 | 0.181 | 0.183 | 11.04 | 11.38 |
| >200d | 0.096 | 0.134 | 0.143 | 0.194 | 0.186 | 0.204 | 0.211 | 0.215 | 20.64 | 18.96 |
| capped | 0.090 | 0.124 | 0.132 | 0.188 | 0.180 | 0.197 | 0.206 | 0.209 | 2.15 | 1.94 |
| overall | 0.128 | 0.203 | 0.220 | 0.379 | 0.358 | 0.402 | 0.428 | 0.456 | | |

- **Blank state == LUO zero, exactly, on every split.** The overall blank share equals the `it_hist_all_luo == 0`
  share (0.128 / 0.379 / 0.456), and P(column zero | LUO == 0) = 1.000 for all 8 columns on train, val and private.
  Blanking all 8 together therefore produces exactly the state a stale item has at inference.
- The excess is a young-listing phenomenon: under 7 days the private blank rate is 0.85-0.98 vs 0.19-0.58 on train
  random-like rows; above 30 days it is 0.17-0.21 vs 0.12-0.14.
- Alternatives computed for documentation only (not run): p vs all training rows (0.972 … 0.131; higher on old
  items, where much of the gap is production-feed selection of popular items, not staleness) and a val target
  (0.861 / 0.828 / … / 0.074).

## 2. Implementation (nn_x/solvers only; nn/, src/, data/feats/, preds/ untouched)

- `nn_x/solvers/agedrop.py` (new): presets, the torch age-bucket function, and `AgeDrop` (in-place blanking of the
  ID-based xt columns of valid candidate rows, dedicated `torch.Generator` seeded with seed + 7919, per-bucket
  accounting for the log).
- `nn_x/solvers/train.py`: 5 additive hunks (backup of the pre-edit file:
  `scratchpad/round5/agedrop/train_pre_agedrop.py`, md5 dd5d0a84…):
  - `from agedrop import AgeDrop`;
  - `--agedrop` (default "");
  - `agedrop = AgeDrop(...) if a.agedrop else None`;
  - `if agedrop is not None: agedrop(b)` between `tr.gather(...)` and `model.dense(b)` (so it acts on regular and
    cooldown-pool batches alike, and the in-model rank/z channels are computed on the blanked values);
  - `log(agedrop.report())` at each eval.
  With the flag off no AgeDrop object exists, nothing else runs and no RNG is drawn. The flag is kept out of
  `model.cfg` (it lives only in `args`), so the checkpoints load in the stock `model.py`: predict, holdout_eval, the
  fine-tune, and the `solvers/stale` replay.
- One bug was found by the unit test before any run: `b["cf"]` stacks cand_f1..f5, so f4 is column 3, not 4. The first
  draft read f5. Fixed (`F4_COL = 3`), plus test T4b, which ties the batch column to `data/flat/val/cand_f4.npy`.

## 3. CPU unit test (`CUDA_VISIBLE_DEVICES="" python nn_x/solvers/test_agedrop.py`: ALL PASS)

Uses the real `sx_base_s23` checkpoint (cfg + weights) and real val batches on the CPU. Log:
`scratchpad/round5/agedrop/test_agedrop.log`.

| test | result |
|---|---|
| T1a: train.py builds AgeDrop only when `--agedrop` is set and calls it only when it is built (source check) | PASS |
| T1b: p ≡ 0 preset vs no call, 3 batches, train mode (dropout on): dense X, logits and full training loss (listwise + K9 aux) bitwise equal; the model RNG stream is untouched | PASS (losses 4.786931 / 5.200158 / 3.979430 on both sides) |
| T1c: baseline loss reproducible | PASS |
| T2: p ≡ 1 blanking vs a batch whose 8 raw ID inputs are NaN on disk (and separately 0 on disk) through attach_xt: dense features, incl. the in-model rank/z channels, are equal on all valid rows | PASS (max abs diff 0.0) |
| T3: non-ID columns, padded rows and other batch tensors bitwise unchanged; dropped rows have all 8 ID columns = 0.0 | PASS |
| T4: torch age buckets == numpy reference on all 1,389,971 val rows; T4b: `cf[..., 3]` is cand_f4; T4c: all 13 buckets populated | PASS |
| T5: realised drop rate per bucket == p (e.g. 0-6h 0.961/0.961, 7-14d 0.578/0.578, capped 0.099/0.097) | PASS |
| T6: model.cfg unchanged by the hook | PASS |

On-GPU confirmation from the treatment logs (first 1000 steps of s23): training-row blank share 0.1284 → 0.4366,
changed 0.3082 (expected 0.128 → 0.437, 0.308). By bucket after blanking: 0-6h 0.977, 6-24h 0.951, 1-2d 0.915,
2-3d 0.883, 3-5d 0.856, 5-7d 0.832, 7-14d 0.620, 14-30d 0.237, 30-60d 0.128, 60-100d 0.121, 100-200d 0.136,
>200d 0.181, capped 0.179, equal to the expected values in section 1.

## 4. Run log

- 23:15 UTC: first launch of both lanes. Both died with CUDA OOM while loading data: the lead's two long runs
  (v23_gru / v24_din) plus two transient holdout_eval jobs started on GPU N at the same moment. No training step had
  run. Logs were renamed `nn_x/solvers/logs/failed_oom_2315_*`.
- 23:17 / 23:20 UTC: relaunched with staggered starts (s23, then s24 once s23 had loaded). PIDs are in
  `nn_x/solvers/logs/RUNNING_agedrop.txt`. GPU N peaked at ~242 of [n] GB. About 3.4 min per 1000 steps, because 4
  trainers share the GPU (the baseline ran at ~2.5 min).
- 00:22 UTC: both runs, test preds and holdout evals are done. Each trained in 61 min (baseline 49 min). Both select
  step 19000 of 19670 on the holdout (a tie with the last step).

## 5. Results (pre-declared yardsticks; train-only; 2 seed pairs averaged per feed)

**Validity:**
- Args differ from the control only in `agedrop` (and `variant`): diffed from `logs/sx_{base,agedrop}_s{23,24}.json`.
- ckpt `cfg` is identical to the control's for both seeds.
- Stale replay check (d) corr: 0.9999920 (s23) and 0.9999920 (s24) for the new checkpoints, PASS; checks a/b/d pass.
- The logged realised blank shares match the targets (section 3). The A/B is valid.

Single runs (selected step, `nn_x/solvers/logs/sx_eval_base_agedrop.txt`):

| run | val | holdout |
|---|---|---|
| sx_base_s23 | 0.39375 | 0.31783 |
| sx_agedrop_s23 | 0.39328 | 0.31618 |
| sx_base_s24 | 0.39288 | 0.31755 |
| sx_agedrop_s24 | 0.39345 | 0.31543 |

**Pre-declared readouts (treatment − control):**

| yardstick | pooled delta ± SE (z) | per seed s23 / s24 | rule | result |
|---|---|---|---|---|
| **stale-val** (CPU2 `both_pre2both`, PRIMARY) | **+0.00089 ± 0.00062 (z 1.44)** | +0.00092 / +0.00085 | > 2 SE | **FAIL** |
| val (stored train-only preds) | +0.00005 ± 0.00061 (z 0.08) | −0.00048 / +0.00057 | ≥ −1 SE | pass |
| holdout | **−0.00189 ± 0.00034 (z −5.58)** | −0.00166 / −0.00212 | ≥ −1 SE | **FAIL** |

**VERDICT: FAIL.** Two of the three pre-declared conditions fail, so the 10-epoch spec is not written (section 6).

Slices:

| slice | val | stale-val | DiD (stale − base replay) |
|---|---|---|---|
| all | +0.00005 ± 0.00061 | +0.00089 ± 0.00062 | **+0.00121 ± 0.00047 (z 2.6)** |
| M<50 (n 5,893) | +0.00058 ± 0.00113 | **+0.00316 ± 0.00116** | +0.00316 ± 0.00091 |
| M>=50 (n 8,530) | −0.00032 ± 0.00068 | −0.00068 ± 0.00066 | −0.00013 ± 0.00050 |
| Tue (n 7,543) | +0.00121 ± 0.00086 | +0.00188 ± 0.00086 | +0.00129 ± 0.00065 |
| Thu (n 6,880) | −0.00123 ± 0.00087 | −0.00020 ± 0.00089 | +0.00113 ± 0.00069 |
| all-cold-positive (stale def., n 4,836) | | +0.01496 ± 0.00118 | +0.00210 ± 0.00094 |
| all-warm-positive (stale def., n 7,424) | | −0.00837 ± 0.00086 | +0.00084 ± 0.00065 |

Holdout slices: all −0.00189 ± 0.00034. The late 20% (feeds nearest the train end, which have the fewest dense
future snapshots) is +0.00031 ± 0.00075.

Stale cost per model (stale replay − base replay; source
`solvers/out/CPU2_stale_replay_pre2s_and_arms/phaseC_agedrop.json`):

| | s23 | s24 |
|---|---|---|
| sx_base | −0.00165 | −0.00043 |
| sx_agedrop | −0.00016 | +0.00051 |

**Information only, not pre-declared** (`scratchpad/round5/agedrop/eval_extra.py`, `eval_extra.log`). Slices by the
age of the feed's positives (young = age(f4) < 7 d):

| slice | n | val | stale-val |
|---|---|---|---|
| all positives young | 2,418 | **+0.02473 ± 0.00187** | +0.02275 ± 0.00176 |
| mixed | 1,613 | +0.00377 ± 0.00121 | +0.00431 ± 0.00114 |
| no young positive | 10,392 | **−0.00627 ± 0.00069** | −0.00473 ± 0.00072 |
| feed young share ≥ 0.3 | 6,183 | −0.00013 ± 0.00099 | +0.00157 ± 0.00100 |

Ensemble readout (info, sx_eval): 1 base + 1 agedrop vs 2 base gives val +0.00031 ± 0.00043 and holdout
−0.00050 ± 0.00024.

### Reading
- **The mechanism works as designed, but it is a young/old trade, not a net gain.**
  - The agedrop models are immune to the stale emulation: stale cost −0.0002 / +0.0005 vs −0.0017 / −0.0004 for the
    control, DiD +0.00121 ± 0.00047.
  - They rank feeds with young positives far better (+0.025 on val).
  - They pay on the 72% of feeds with no young positive (−0.0063): the training-time noise in the popularity inputs
    weakens the warm ranking.
  - At val's mix that nets to 0. At the stale mix it is +0.0009 (n.s.). On the warm, dense-count holdout it is −0.0019.
  - This is the same shape as the GBDT-level `it_hist_hz` result (featnew_it_hist_hz.md: young-positive feeds
    +0.04, no-young-positive feeds −0.006) and the uniform `--xt_drop q15` result. Making train-row popularity
    semantics match the test only trades cold for warm.
- **The holdout loss is mostly the dense-future-count artifact.** It disappears on the late-20% holdout slice
  (+0.0003 ± 0.0008), where future train snapshots are fewest. That does not change the pre-declared verdict.
- **The gain sits on M<50 and Tue** (stale +0.0032 and +0.0019). M>=50 and Thu are ≤ 0 on stale-val.
- The 2-epoch harness and these p values may understate the 10-epoch effect, but nothing here justifies a 10-epoch
  family under the rules.

## 6. Decision and recommendation for the lead
- **FAIL under the pre-declared rule.** No 10-epoch family spec is issued; nothing is launched.
- Do not add agedrop members to the final ensemble candidates. The checkpoints `sx_agedrop_s{23,24}` exist for
  reference only. Their test preds are in `nn_x/solvers/preds/` (not copied to `preds/`).
- The val fine-tune half (0.5 z(ft) + 0.5 z(train-only)) already re-calibrates cold items on exploration labels.
  That remains the right staleness lever: it helps more on colder data (review_ideas 1.3), without the warm cost.
- If anyone revisits the idea, the only untested direction with a plausible net gain is a much milder,
  young-only variant (e.g. p only below 14 days, at val-target rates). It would need a new pre-declaration. I do
  not recommend spending GPU on it: three independent constructions (it_hist_hz, uniform xt_drop, this one) all
  show the same cold/warm trade.

## 7. Caveats
- **bf16 vs fp32 inference noise.** The CPU replay's base delta (−0.00033) differs from the stored GPU-preds val delta
  (+0.00005) by 0.0004. There are no exact ties in either. Per-model NDCG moves by up to 0.00035 between bf16 GPU and
  fp32 CPU inference (sx_base_s24 0.39288 → 0.39318, sx_agedrop_s24 0.39345 → 0.39310).
  - The pre-declared stale yardstick compares replays with replays, consistently.
  - Read on the stored basis (per feed: stored val delta + replay DiD), it is +0.00126 ± 0.00063 (z 1.99): still not
    > 2 SE. The holdout condition also fails at z −5.6, so the verdict does not depend on this.
- The pre2 ID part of `pre2_fe_s` / `pre2_fe_z` is not blanked together with the co-vis inputs. CPU2 measured that
  part's staleness effect at 0.0000-0.0001. History-side co-vis staleness is not emulated, as in CPU2.
- Stale-val is slightly harsher than private on young buckets (5-7 d: 0.915 vs 0.847 zero share;
  `scratchpad/round5/agedrop/stale_coverage_by_age.txt`). The p values target private.
- 2-epoch runs, 2 seeds, one arm. Per-model stale costs vary a lot between seeds (as in CPU2), which is why only
  paired, seed-pooled deltas are read.

## 8. Files and commands
- Code: `nn_x/solvers/agedrop.py`, `nn_x/solvers/train.py` (flag `--agedrop`), `nn_x/solvers/test_agedrop.py`.
- Runs: `nn_x/solvers/ckpt/sx_agedrop_s{23,24}.pt`; `nn_x/solvers/preds/nn_sx_agedrop_s{23,24}_trainonly_{val,test}.npy`;
  `nn_x/solvers/preds/holdout/nn_sx_agedrop_s{23,24}_holdout.npy`; logs `nn_x/solvers/logs/sx_agedrop_s{23,24}.{log,json,stdout}`;
  `nn_x/solvers/logs/sx_eval_base_agedrop.{txt,json}`.
- Stale: `solvers/out/CPU2_stale_replay_pre2s_and_arms/phaseC_agedrop.{json,log}`; replays cached in
  `solvers/out/stale/replay/cf_sx_agedrop_s{23,24}_{base,both_pre2both}.npy`.
- Scratch: `/tmp/scratchpad/round5/agedrop/`
  (measure.py/.json/.log, eval_extra.py/.json/.log, test_agedrop.log, phaseC_agedrop.stdout,
  stale_coverage_by_age.txt, train_pre_agedrop.py).
- Reproduce:
  ```
  cd <repo>
  CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/solvers/test_agedrop.py
  bash nn_x/solvers/run_sx_lane.sh agedrop "--agedrop prv_rl13" 23   # and 24 (GPU N)
  CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/solvers/sx_eval.py base agedrop
  CUDA_VISIBLE_DEVICES="" .venv/bin/python solvers/CPU2_stale_replay_pre2s_and_arms.py C sx_base_s23:sx_agedrop_s23,sx_base_s24:sx_agedrop_s24 phaseC_agedrop
  ```
