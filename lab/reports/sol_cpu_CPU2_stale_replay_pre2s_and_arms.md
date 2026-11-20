# CPU2: stale-val replay of pre2s and the sx GPU arms (Phases A, B, C done)

Code: `solvers/CPU2_stale_replay_pre2s_and_arms.py` (phases `prep`, `A`, `A_full`, `B`, `C`), replay copy in
`solvers/stale/{run,data,model}.py`. CPU only (`CUDA_VISIBLE_DEVICES=""`, 16 threads).
Outputs: `solvers/out/stale/cf/` (counterfactual inputs), `solvers/out/stale/replay/` (cached CPU replays, float32,
val cand-aligned), `solvers/out/CPU2_stale_replay_pre2s_and_arms/` (pre-check JSONs, phase JSON/log/stdout).
No test predictions were produced. The task is val-only replay, so nothing is written to preds/.

## TL;DR

- **Phase A: pre2s is NOT staleness-fragile.** At private-like coverage the cvnd_pre2s gain is unchanged or slightly larger:
  - base delta **+0.00189 ± 0.00058**;
  - stale delta **+0.00246 ± 0.00057** (both+pre2both);
  - diff-in-diff **+0.00057 ± 0.00044**.
  
  Rule (i) does not flag, so no action follows (no raw-channel ablation). The decision is identical under both-only (+0.00059 ± 0.00044) and under a harsher emulation of the pre2 'new' head too (+0.00059 ± 0.00044).
- Mechanism (verified): the stale scenario costs the cvnd baselines −0.0021 / −0.0014 and the pre2s models −0.0009 / −0.0014. pre2 content scores partly replace the popularity signal that staleness removes.
  - The ID part of pre2 that staleness would remove is small for newly cold items: mean |id_contrib| 0.71 vs 1.26 on items that stay warm.
  - Removing it costs the pre2s models only 0.00000-0.00005.
- Pre-checks:
  - (a), (b) and (d) PASS.
  - (b) passes only with the faithful builder, i.e. the per-config mean of within-feed z, which reproduces the live pre2_fe_z exactly. The naive z of the config-mean pre2_fe is off by up to 0.37.
  - (c): P(id_contrib==0 | LUO==0) = 1.000 and P(id_contrib==0 | LUO>0) = 0.272. ID coverage lies entirely inside LUO coverage, so the directional overlap is 100%. The symmetric Jaccard is 0.728, and private has the same structure (Jaccard 0.739).
  - (c) is reported under both readings. The both-only scenario gives the same decision, so the ambiguity does not matter.
- **Phase B: no flag.** F7 (v17_pre2s_s101..106) vs F6 family train-only z-means:
  - base **+0.00147 ± 0.00074**;
  - stale **+0.00170 ± 0.00072**;
  - diff-in-diff **+0.00023 ± 0.00055**.
  
  A pool containing F7 is therefore not restricted to pick 2 by rule (i). Caveat: F7 vs F6 loses on all-cold-positive feeds (−0.0033 ± 0.0014 at val coverage), which the 3-epoch pre2s pair does not. The likely cause is the recipe difference (lr 2e-3), not pre2s (suspected).
- **Supplementary pools (TO halves, not pre-declared):** F5+F6 -> F5+F6+F7 gives base +0.00057 ± 0.00035 and stale **+0.00111 ± 0.00034**.
- **Phase C (sx_* K9 arms, seeds 23/24 pooled): no veto (rule ii).** Neither arm is a winner on val anyway.
  - tau30: base −0.00111 ± 0.00049 (worse), stale −0.00077 ± 0.00049.
  - pl20: base +0.00027 ± 0.00045 (null), stale −0.00037 ± 0.00045, diff-in-diff −0.00064 ± 0.00039. It leans on popularity a little more, consistent with the rationale, but below 2 SE.

## 1. Setup and hygiene

- `solvers/stale/` is a copy of `solvers/errors/nncpu/{run.py,data.py,model.py}`. `model.py` is byte-identical to `nn/model.py`; `diff` is empty.
- `data.py`:
  - `feat_path` now raises `FileNotFoundError` if an override or feature file is missing.
  - The `os.link` / `shutil.copy2` fallback (old `_orig_feat_path_unused`, nncpu/data.py:43-54), which could write into `nn/feat_cache`, was deleted, along with `import shutil`.
  - A grep for `os.link|copy2|np.save|torch.save|mkdir` in the copy finds no write path.
- `run.py`:
  - Checkpoints are resolved read-only over `[nn/ckpt, nn_x/solvers/ckpt]` (or `STALE_CKDIRS`); `ckpt_path` raises if a name is missing or ambiguous.
  - `run()` asserts that `max_hist` matches the checkpoint and resets `d.xt` per model.
  - The `__main__` output goes to `solvers/out/stale/replay/`.
- Replays read `nn/feat_cache/val/*` and `data/feats/val/*`; overrides come from the review_shift `both_*` files and `solvers/out/stale/cf/*`.

## 2. Counterfactual inputs

| scenario | what changes (val) |
|---|---|
| base | live inputs |
| both | review_shift `both_*` (8 files: it_hist_{all,et1-4}_luo, seq_cv_{invd20,norm,pop}): train sources cut at Sep 17 and no test snapshots. Cold share 0.379 -> 0.485 (private 0.456; pessimistic) |
| both+pre2both (**primary**) | both + `newly_cold = (both_it_hist_all_luo == 0) & (val it_hist_all_luo > 0)` (10.56% of rows); `pre2_fe' = where(newly_cold, pre2_fe_noid, pre2_fe)`; `pre2_fe_z'` = per-config mean of within-feed z (the live builder, see check b); `pre2_fe_s' = (pre2_fe' + 20) / 6` |
| both+pre2full (sensitivity, not pre-declared) | both+pre2both + the 'new' head: per config `new' = where(newly_cold, new_noid, new)` aligned with the new head's val->test map; `pre2_new_z'`, `pre2_new_s'` rebuilt the same way |

- **Un-emulated in the primary scenario:** pre2_new, pre2_new_z, pre2_new_s (covered by the pre2full sensitivity), pre2_hmax32, and all ctx_* inputs. History-side co-vis staleness is also left out (review_shift: seq_cv_invd20 zero rate 0.890 vs private 0.902).
- has:pre2_fe is unchanged: pre2_fe has no NaN on val.
- **Rows changed by pre2both:**
  - 146,726 rows are newly cold. Only 43,419 of them have an item ID (29.6%); on the other 103K the change is the P5 alignment residual, at most 0.009.
  - Mean shift on the changed rows: pre2_fe −0.155, pre2_fe_z −0.040. On the ID rows alone the shift is about −0.52 in fe units.
  - pre2full changes the same 43,419 rows of the new head by −0.47 (z −0.13).
- **ID coverage after emulation:** the ID-less share becomes 0.579 on val (live val 0.55, private 0.598). The ID channel is therefore emulated slightly short of the private level, while LUO coverage overshoots it (0.485 vs 0.456).
- Per-config construction: `pre2_*` val = `assemble.py --val v3At2,v4At2 --calib v3t2,v4t2 --prefix pre2_` (pretrain_report section 8c). It was reproduced from `preds/pre_{v3At2,v4At2,v3t2,v4t2}_*_val.npy` with `assemble.affine`, with max abs diff 0.0 for pre2_fe, pre2_fe_noid, pre2_fe_z, pre2_new and pre2_new_z.

## 3. Pre-checks (`prechecks_abc.json`, `prechecks_pre2full.json`, phase JSONs)

| check | result | verdict |
|---|---|---|
| (a) live pre2_fe_s == (pre2_fe + 20)/6 | max abs diff 0.0 (pre2_new_s also 0.0) | PASS |
| (b) z builder reproduces live pre2_fe_z | naive z(pre2_fe): max 0.367 (FAILS). The live z is the mean over the 2 configs of the within-feed z (pretrain/assemble.py:53 per config, :70 mean over configs). The per-config builder reproduces pre2_fe_z with max abs diff **0.0**, and is the one used for pre2_fe_z' | PASS (faithful builder) |
| (c) P(id_contrib==0 \| LUO==0) vs P(id_contrib==0 \| LUO>0) | **1.000** vs **0.272**. P(LUO>0 \| has ID) = 1.000; Jaccard(ID, LUO) = 0.728. Same on test: private 1.000 / 0.262, Jaccard 0.739 | directional PASS (100%); symmetric reading 0.73 < 0.80 -> both-only also reported, same decision |
| (d) base replay corr vs stored preds/nn_<m>_trainonly_val | v8_nd_e3 0.9999962, v18_pre2s_e3_s23 0.9999964, v8_nd_e3_s42 0.9999965, v18_pre2s_e3_s42 0.9999964 (all >= 0.99999). Max abs diff 0.30-0.40 logits, at isolated rows | PASS |

- On (c): the encoder's ID vocabulary is a strict subset of LUO coverage on every split, so "LUO-cold => ID-less" holds exactly. That is the only implication the emulation uses.
- About 27% of LUO-warm items also lack an ID. Some items that keep a small LUO count under staleness would also lose their ID, and this is not emulated (conservative).
- The replayed base deltas reproduce the stored-file W2 numbers: +0.00189 ± 0.00058 here vs +0.00191 ± 0.00058 (CURRENT_STATUS 3l, 18:30).
- v8_nd_e3's stale cost (−0.00208) reproduces review_shift's "both, 3-epoch v8_nd_e3 −0.0021".

## 4. Phase A: cvnd (v8_nd_e3, _s42) -> cvnd_pre2s (v18_pre2s_e3_s23, _s42), 3 epochs, train-only val

Paired per feed, the 2 pairs averaged per feed before the SE; n = 14,423 feeds. Source: `phaseA.json`, `phaseA_sens_pre2full.json`.

| scenario | base delta | stale delta | diff-in-diff | rule (i) flag |
|---|---|---|---|---|
| both | +0.00189 ± 0.00058 | +0.00248 ± 0.00057 | +0.00059 ± 0.00044 | no |
| **both+pre2both (primary)** | +0.00189 ± 0.00058 | **+0.00246 ± 0.00057** | **+0.00057 ± 0.00044** | **no** |
| both+pre2full (sensitivity) | +0.00189 ± 0.00058 | +0.00248 ± 0.00057 | +0.00059 ± 0.00044 | no |

Per pair (base / both / both+pre2both):

| pair | base | both | both+pre2both |
|---|---|---|---|
| v8_nd_e3 -> v18_pre2s_e3_s23 | +0.00139 | +0.00254 | +0.00254 |
| v8_nd_e3_s42 -> v18_pre2s_e3_s42 | +0.00239 | +0.00243 | +0.00238 |

Stale cost per model (scenario minus base, own replay):

| model | both | both+pre2both |
|---|---|---|
| v8_nd_e3 | −0.00208 | −0.00208 |
| v8_nd_e3_s42 | −0.00138 | −0.00138 |
| v18_pre2s_e3_s23 | −0.00094 | −0.00094 |
| v18_pre2s_e3_s42 | −0.00134 | −0.00139 |

Slices (primary scenario both+pre2both; cold = it_hist_all_luo == 0 under the scenario's own popularity):

| slice | n (base / stale def.) | base delta | stale delta | diff-in-diff |
|---|---|---|---|---|
| Tue | 7,543 | +0.00259 ± 0.00082 | +0.00300 ± 0.00080 | +0.00041 ± 0.00062 |
| Thu | 6,880 | +0.00113 ± 0.00083 | +0.00186 ± 0.00080 | +0.00074 ± 0.00061 |
| M<50 | 5,893 | +0.00363 ± 0.00111 | +0.00477 ± 0.00108 | +0.00114 ± 0.00085 |
| M>=50 | 8,530 | +0.00069 ± 0.00062 | +0.00086 ± 0.00060 | +0.00017 ± 0.00045 |
| all-cold-positive | 3,422 (base) / 4,836 (stale) | +0.00412 ± 0.00110 (base def.) | +0.00154 ± 0.00093 | −0.00102 ± 0.00073 (stale def.); −0.00219 ± 0.00075 (base def.) |
| all-warm-positive | 9,118 / 7,424 | +0.00102 ± 0.00079 (base def.) | +0.00304 ± 0.00087 | +0.00136 ± 0.00067 (stale def.); +0.00163 ± 0.00061 (base def.) |

Reading:

- **Decision (rule i, pre-declared):** diff-in-diff +0.00057 ± 0.00044 is not < −2 SE, and the stale delta is > 0. **No flag -> no action.** pre2s stays as is for nn_final7; the raw-channel ablation is not ordered.
- Note for the lead: in cvnd_pre2s, pre2_fe_s / pre2_new_s / pre2_hmax32 are already XT_RELONLY (nn/train.py:51-52). The model never sees their raw scale, only the in-model within-feed rank/z (model.py dense_features, lines 123-130). The 'raw-channel' ablation would therefore drop those in-model rank/z channels.
- **Where the gain moves under staleness:** the pre2s advantage shrinks on feeds whose positives are already cold at val coverage (DiD −0.0022 ± 0.0008, base-defined) and grows on all-warm-positive feeds (+0.0016 ± 0.0006).
  - Under staleness many negatives in cold-positive feeds also lose popularity. That helps the cvnd baseline more than the pre2s model, which already ranked cold items by content.
  - On warm-positive feeds, the baseline loses its popularity contrast while pre2s keeps a content signal.
  - The net effect is positive. The gain is still concentrated on M<50 (stale +0.0048 vs +0.0009 on M>=50), and Thu/Tue are both positive.
- **The pre2-specific emulation is nearly inert:** both -> both+pre2both changes the pooled stale delta by −0.00002, and adding the new head by +0.00002.
  - Newly cold items carry small ID contributions: mean |id_contrib| 0.71 vs 1.26 for items that stay warm.
  - pre2s fragility from losing IDs is therefore bounded at about 0.0000-0.0001 on this yardstick.
- Rule (ii) is not applicable to Phase A. Its computed value (no veto) is in the JSON for completeness.

## 5. Phase B: F7 (v17_pre2s_s101..106) vs F6 (v13_lr1_mix1, v14_s92..96), family TO z-means

Run 21:07-21:14 UTC, as soon as all six F7 stage-1 checkpoints and train-only preds existed. Each family = equal-weight mean of the members' within-feed z; one family pair; per-feed SE.
Source: `phaseB.json`, `phaseB.stdout`.

- **Check (d):**
  - F7: s101 0.9999992, s102 0.9999991, s103 0.9999988, s104 0.9999992, s105 0.9999991, s106 0.9999991.
  - F6: 0.9999983-0.9999989.
  - All PASS.
  - F6 members read no pre2 input (asserted on `xt_names`), so their both+pre2both replay is the both replay. The alias is exact: the Phase A v8 replays were identical under both scenarios.
- **Replay levels (TO z-mean):**
  - F6: base 0.39706 -> stale 0.39520 (cost −0.00186).
  - F7: base 0.39853 -> stale 0.39690 (cost −0.00163).

| scenario | base delta (F7 − F6) | stale delta | diff-in-diff | rule (i) flag |
|---|---|---|---|---|
| both | +0.00147 ± 0.00074 | +0.00170 ± 0.00072 | +0.00023 ± 0.00055 | no |
| **both+pre2both (primary)** | **+0.00147 ± 0.00074** | **+0.00170 ± 0.00072** | **+0.00023 ± 0.00055** | **no** |

Slices (primary scenario):

| slice | base delta | stale delta | diff-in-diff |
|---|---|---|---|
| Tue | +0.00182 ± 0.00104 | +0.00237 ± 0.00102 | +0.00055 ± 0.00077 |
| Thu | +0.00109 ± 0.00104 | +0.00096 ± 0.00102 | −0.00012 ± 0.00077 |
| M<50 | +0.00308 ± 0.00139 | +0.00366 ± 0.00138 | +0.00058 ± 0.00102 |
| M>=50 | +0.00036 ± 0.00079 | +0.00034 ± 0.00077 | −0.00001 ± 0.00060 |
| all-cold-positive (base def. n=3,422 / stale def. n=4,836) | **−0.00330 ± 0.00140** | −0.00188 ± 0.00122 | −0.00060 ± 0.00095 (stale def.); +0.00028 ± 0.00108 (base def.) |
| all-warm-positive (9,118 / 7,424) | +0.00336 ± 0.00100 | +0.00391 ± 0.00110 | +0.00062 ± 0.00082; +0.00010 ± 0.00074 |

- **Decision (rule i):** no flag. Pools containing F7 are not restricted to pick 2 by this rule.
  - The stale delta is >= 0 (+0.0017), so the condition "stale delta >= 0" in the Phase-B clause holds anyway.
  - The usual final-pick rules apply unchanged.
- **Observation for the lead (verified numbers; cause suspected):** at 10 epochs, F7 vs F6 trades cold for warm.
  - On all-cold-positive feeds: −0.0033 ± 0.0014 (z −2.4) at val coverage, −0.0019 ± 0.0012 at stale coverage. On all-warm-positive feeds: +0.0034.
  - The 3-epoch pre2s pair goes the other way on cold-positive feeds (+0.0041 ± 0.0011, section 4). So pre2s is probably not the cause.
  - F7 also differs from F6 in recipe: lr 2e-3 (ii) vs 1e-3 (iii). That is the likely source, but it is not isolated here.
  - The cold/warm split is not one of the acceptance slices (M<50 / M>=50 / Tue / Thu). The staleness-weighted overall delta stays positive, because the stale scenario already shifts the mix toward cold feeds (33.5% all-cold-positive under both vs 23.7% at val coverage).
- **Not run:** F7 + s107. The 7th seed was at step 57.5k/98k at 21:20 UTC (ETA ~22:25). `python solvers/CPU2_stale_replay_pre2s_and_arms.py B` reruns Phase B once its preds exist. It reuses the cached replays (about 3 min) and adds the "F7(+s107) vs F6" readout under `extra`.

### 5b. Supplementary (not pre-declared, information only): final-ensemble candidate sets, TO halves

Pool = equal-weight mean over families of the per-feed z of each family's member z-mean. F5 = the six nn_final5 members (v12_all_e10/s81/s82, v12_allprew_s83-85). Under the stale scenarios the prew members get the review_shift `prewboth_*` masking, exactly as `cf_pool.py` did.

- This replay pipeline equals the earlier cf_pool replays: corr 1.0000000 for all 14 F5 / v13_lr1_mix1 base and both files.
- The F5 base level 0.39691 equals sol_errors section 3.5 S0.

| comparison | base delta | stale delta | diff-in-diff | stale M<50 / M>=50 | stale Tue / Thu |
|---|---|---|---|---|---|
| F5 -> F5+F6 | +0.00054 ± 0.00045 | +0.00039 ± 0.00046 | −0.00016 ± 0.00045 | +0.0003 / +0.0004 | +0.0001 / +0.0007 |
| F5+F6 -> F5+F6+F7 | +0.00057 ± 0.00035 | **+0.00111 ± 0.00034** | +0.00055 ± 0.00040 | +0.0010 / +0.0012 | +0.0018 / +0.0004 |
| F5 -> F5+F6+F7 | +0.00111 ± 0.00047 | +0.00150 ± 0.00049 | +0.00039 ± 0.00047 | +0.0013 / +0.0017 | +0.0019 / +0.0011 |
| F5 -> F7 | +0.00162 ± 0.00067 | +0.00148 ± 0.00070 | −0.00014 ± 0.00053 | +0.0025 / +0.0008 | +0.0015 / +0.0014 |

- On TO halves, adding F7 to the pool is at least as good at private-like coverage as at val coverage.
- This covers TO halves only. The actual pick uses 0.5 fta + 0.5 TO family mixes and the holdout rule.

## 6. Phase C: sx_* GPU arms (K9 tuning, 2-epoch cvnd_pre2s base, `nn_x/solvers/run_sx_lane.sh`)

Run 21:41-21:45 UTC. Six checkpoints existed then: sx_{base,tau30,pl20}_s{23,24}. The task said 8, but the remaining pair (arm (c), `--aux_pos 1.0` on the winner) is conditional and had not been launched.

- Scenarios: base and both+pre2both, as specified.
- Check (d) against `nn_x/solvers/preds/nn_<m>_trainonly_val.npy`:
  - s23: base 0.9999919, pl20 0.9999924, tau30 0.9999918.
  - s24: base 0.9999916, pl20 0.9999915, tau30 0.9999907.
  - All >= 0.99999, so PASS. These are about 10x further from 1 than the nn/ members, but still inside the threshold.
- Source: `phaseC_tau30.json`, `phaseC_pl20.json`. The seed-23-only previews are `phaseC_partial_*_s23.json`.

Pooled over seeds 23/24, train-only val, paired per feed:

| arm vs sx_base | base delta | stale delta | diff-in-diff | per seed (base / stale) | rule (ii) veto (stale < −2 SE) |
|---|---|---|---|---|---|
| tau30 (`--aux_tau 30`) | **−0.00111 ± 0.00049** | −0.00077 ± 0.00049 | +0.00034 ± 0.00040 | s23 −0.00090 / +0.00024; s24 −0.00132 / −0.00179 | no (−0.00077 > −0.00098), and it is not a winner |
| pl20 (`--aux_mode pl --aux_pl_k 20`) | +0.00027 ± 0.00045 | −0.00037 ± 0.00045 | −0.00064 ± 0.00039 | s23 +0.00115 / +0.00040; s24 −0.00060 / −0.00113 | no (−0.00037 > −0.00090); not a winner on base |

Slices:

| arm | slice | base delta | stale delta | diff-in-diff |
|---|---|---|---|---|
| tau30 | Tue / Thu | −0.00023 / −0.00208 ± 0.00070 | +0.00027 / −0.00192 ± 0.00073 | +0.00050 / +0.00016 |
| tau30 | M<50 / M>=50 | −0.00115 / −0.00109 | −0.00080 / −0.00076 | +0.00035 / +0.00033 |
| tau30 | all-cold-pos / all-warm-pos (stale def.) | −0.00145 / −0.00075 | −0.00052 / −0.00076 | +0.00093 ± 0.00069 / −0.00002 |
| pl20 | Tue / Thu | +0.00055 / −0.00003 | +0.00013 / −0.00091 | −0.00042 / −0.00088 |
| pl20 | M<50 / M>=50 | +0.00192 ± 0.00087 / −0.00086 ± 0.00047 | +0.00044 / −0.00092 ± 0.00047 | **−0.00148 ± 0.00075** / −0.00006 |
| pl20 | all-cold-pos / all-warm-pos (stale def.) | +0.00066 / +0.00013 | +0.00009 / −0.00044 | −0.00056 / −0.00057 |

Stale cost per model (both+pre2both minus base):

| model | s23 | s24 |
|---|---|---|
| sx_base | −0.00165 | −0.00043 |
| tau30 | −0.00051 | −0.00089 |
| pl20 | −0.00240 | −0.00095 |

The per-model stale cost varies a lot between seeds (base −0.0017 vs −0.0004), which is why only paired, seed-pooled deltas are read.

Reading:

- **Rule (ii):** no veto for either arm. Neither arm is a winner on base val, though: tau30 is worse at z −2.3, and pl20 is null at +0.6 SE. The veto question is moot.
- pl20 is the one arm whose gain shrinks under staleness:
  - diff-in-diff −0.00064 ± 0.00039 (z −1.6), with its M<50 gain shrinking by −0.0015 ± 0.0008;
  - both seeds point the same way (s23 −0.00075, s24 −0.00053).
  
  This is consistent with the rationale ("K9 imitation of the fresh-coverage production ranker may lean harder on popularity"), but below 2 SE.
- If a PL-based arm (c) is launched, replay it with the commands in section 8. It would be vetoed only if its pooled stale delta is < −2 SE.

## 7. Decisions (pre-declared rules applied exactly)

| item | rule | numbers | outcome |
|---|---|---|---|
| Phase A pre2s (3-epoch pairs) | (i) DiD < −2 SE and stale < 0 | DiD +0.00057 ± 0.00044, stale +0.00246 | **no flag -> no action** (no raw-channel ablation) |
| Phase B F7 vs F6 | (i) | DiD +0.00023 ± 0.00055, stale +0.00170 | **no flag -> F7 pools eligible for pick 1** under the usual rules |
| Phase C tau30 | (ii) stale < −2 SE, pooled over 23/24 | stale −0.00077 ± 0.00049 | no veto (not a winner: base −0.00111 ± 0.00049) |
| Phase C pl20 | (ii) | stale −0.00037 ± 0.00045 | no veto (not a winner: base +0.00027 ± 0.00045) |
| validity | (a), (b), (d) must pass | all pass; (c) directional pass / symmetric 0.73 | valid. both-only gives identical decisions in A and B (Phase C ran both+pre2both only, per spec) |

## 8. How to rerun / extend

```
cd <repo>; export CUDA_VISIBLE_DEVICES=""
.venv/bin/python solvers/CPU2_stale_replay_pre2s_and_arms.py prep            # cf files + checks a,b,c (seconds)
.venv/bin/python solvers/CPU2_stale_replay_pre2s_and_arms.py A               # Phase A (cached)
.venv/bin/python solvers/CPU2_stale_replay_pre2s_and_arms.py B               # Phase B; adds F7+s107 once v17_pre2s_s107 preds exist
.venv/bin/python solvers/CPU2_stale_replay_pre2s_and_arms.py C sx_base_s23:sx_<arm>_s23,sx_base_s24:sx_<arm>_s24 phaseC_<arm>
```

- About 25 s (3-epoch) to 55 s (10-epoch retr16) per model per scenario on 16 threads.
- Replays are cached in `solvers/out/stale/replay/cf_<model>_<scenario>.npy`.
- Models without pre2 inputs get both+pre2both := both (an exact alias).
- prew models get the `prewboth_*` masking automatically.

## 9. Caveats

- The emulation covers popularity / co-vis coverage plus the pre2 ID part only. pre2_hmax32, ctx_*, the encoder's own content drift and history-side co-vis staleness are not emulated.
- 'both' overshoots private's LUO-zero share (0.485 vs 0.456), so it is a pessimistic stress for popularity reliance.
- Train-only (TO) halves only; the fine-tuned halves are not replayed.
- SEs are per-feed with pairs averaged per feed. Fine-tune and user-level clustering are not involved, and user-clustered SE = iid SE on val (REVIEW_SYNTHESIS).
- Nothing was fit on val, so the nested user-fold CV rule does not apply. All scenario definitions and decision rules were fixed before scoring. pre2full is an added, explicitly non-primary sensitivity.
- No holdout counterpart exists for the stale scenario: the holdout is train feeds with their own coverage. The "late holdout slice" of the unified rule is therefore not part of this yardstick; the holdout numbers for the arms are in the GPU workflow's reports.
- Phase B is one family pair (6 vs 6 members), so its SE does not include seed-to-seed variation of a family. Phase C pools 2 seeds per arm.
- Per-model stale costs vary a lot between seeds (sx_base −0.0017 vs −0.0004), so read only paired deltas.
- The (c) definition (directional vs symmetric) was settled after the coverage numbers were seen. The decision does not depend on it: both-only and both+pre2both agree in Phases A and B.
