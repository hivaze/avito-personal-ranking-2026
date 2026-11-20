# Feature-design review: synthesis (FEATURES_REVIEW)

Date 2026-11-11, ~17:20-18:00 UTC. Synthesis lead of the feature-review workflow. CPU only (`CUDA_VISIBLE_DEVICES=""`, ≤ 16 threads). No file under nn/, src/, pretrain/, data/feats/, preds/ or subs/ was modified.

**Inputs.**
- Workflow reports: `reports/featrev_{prep,inventory,research}.md` and `reports/featnew_{hzcv,ixh,it_hist_hz,ud}.md`, each with the builder's adversarial verdict.
- Concurrent reviews: `reports/review_{leakage,shift,integrity}.md`, `REVIEW_SYNTHESIS.md`.
- Current state: `CURRENT_STATUS.md` up to 17:20, `nn_report.md` up to the W2 entry (l.510-517), and the live `nn/train.py` / `nn/data.py` / `nn/finetune.py`.

**New work in this step.**
- **FR-1** (section 1): a check of the live `cvnd_pre2` preset.
- **FR-9**: the RESEARCH-3 inference-only test.
- Clip-safe `pre2_fe_s` / `pre2_new_s` files, built by `feat_new/pre2_rescale.py` (section 3.2).
- Scratch: `/tmp/scratchpad/featrev/synth/`.

Tags: **[V]** = verified by a command or computation (mine or a cited one); **[S]** = suspected or inferred.

---

## 0. Bottom line

1. **No leak in the live feature path [V].** Every train-stage and fine-tune input of nn_final5 / v11 passed the generic detector, the has-later audit and the code reading:
   - review_leakage TL;DR and "Checked and correct";
   - INVENTORY §2.d and §5;
   - REVIEW_SYNTHESIS §3.

   The only material leak found today, the train-row pre_* content-table bit, is quarantined. The NN loader guardrail is now closed: `nn/feat_cache/train/pre_*` are gone and the `cvnd_pre` / `cvnd_prez` presets were removed (`nn/train.py:37`; nn_report l.488). None of the four new families shows a leak flag.
2. **Scaling is right for almost every input.**
   - Of 104 train-stage inputs, 8 have val z-sd outside 0.75-1.25, all explained (PREP TL;DR 1).
   - Count families get log1p. f1..f5 are stable quantile codes that need nothing.
   - fp16 storage is lossless at 4 decimals of AUC.

   **The one real scaling bug is the ±10 clip on raw columns that enter only through the in-model rank/z.** It affects `prew_fe` / `prew_new` (live in 3 nn_final5 members) and, newly found here, `pre2_fe` / `pre2_new` in the `cvnd_pre2` preset that produced the W2 result (FR-1).
3. **The main structural issue is train-row popularity and co-vis semantics vs stale test rows.** It is real and well localised:
   - young listings for `it_hist_*_luo`;
   - censored zeros for `seq_cv_*`.

   **Five causal or horizon-matched variants have now been tried, and none beats the as-built inputs**: causal −0.0018, 4-day-lagged −0.0027, it_hist_hz ≈ 0, hzcv ≈ 0, ixh ≈ 0. The 14-day train window is the binding constraint (section 3). The mitigation is the val fine-tuned half of the mix, so keep it.
4. **The feature space on disk is saturated relative to the NN (INVENTORY-3) [V].** Over the mix, no single feature and no family beats 2 SE after multiplicity. The only on-disk signal with a ≥ 2 SE residual over the train-only NN is `pre_*`, +0.0010 ± 0.0005. W2 gave a marginal train-stage pass: `cvnd_pre2` vs `cvnd`, +0.00124 ± 0.00061 (z 2.0), all of it in M<50 feeds (nn_report l.510-517).
   - **The highest-value feature action left** is to re-run that preset with the clip-safe `pre2_*_s` columns and without `pre2_id_contrib` (section 3.2).
5. **External data.** The data has no text, price, geo or seller fields, and item and user IDs are anonymised. External enrichment is therefore impossible. The research map (featrev_research §4) shows every applicable technique from comparable competitions is already present or was measured null. The one gap it found, horizon-aligned features (M5 / Favorita), was built as hzcv and fails for the structural reason in point 3.

---

## 1. Confirmed problems, ranked by impact

Ranked by actionable impact on private LB and on pending decisions; FR-3 / FR-4 are the largest mismatches but have no winning fix. "Live" means it touches nn_final5 / v11 members, the running nn_final6 (v14) runs, or the W2 decision. Every fix to nn/*.py is an instruction for the lead; none was applied.

| # | problem | evidence | live? | minimal fix (instruction) |
|---|---|---|---|---|
| **FR-1** | **`cvnd_pre2` repeats the prew clip bug, and adds `pre2_id_contrib` against P5/N3.** `pre2_fe` / `pre2_new` / `pre2_hmax32` / `pre2_id_contrib` are XT_RELONLY (`nn/train.py:43-46`), so they enter only through rank/z of the `attach_xt` column, which is clipped to ±10 (`nn/data.py:199`). | [V] (this step, 3M-row samples per split). Present rows ≥ 10, i.e. tied at 10: pre2_fe **train 92.5% / val 83.8% / test 84.7%**; pre2_new 87.9% / 75.9% / 76.9%. The tie share differs train vs val, which reproduces the PREP-1 spread mismatch. Val per-feed AUC of the rank channel as fed (clip10, fp16) is **0.5521** vs 0.6632 raw; `pre2_fe_z` carries the full ordering (0.6632). `pre2_id_contrib` no-ID (exact 0) share: train random-like (p > 0.9, M ≥ 50, Sep 9+) **0.464** vs val 0.548 / test 0.592; lift(y \| no-ID) 0.814 vs 0.883 (val M ≥ 50 0.889). That is a density skew of the LEAKAGE-6 type, not a label leak. Train pre2 is NaN on all Sep 6-8 rows (27.9%), so `has:pre2_fe` is a pure date flag on train and constant 1 on val/test. CURRENT_STATUS l.187 (16:25) says W2 was to run "without id_contrib", but the preset includes it. | **yes**: W2 (`v16_pre2_e3_s23/s42`, nn/logs) and any pre2 member of nn_final6 | Do not edit `cvnd_pre2` or `attach_xt` in place, because its two checkpoints use it. New preset **`cvnd_pre2s`**, defined in section 3.2, using `pre2_fe_s` / `pre2_new_s` (built, `data/feats_new/*/`) and no `pre2_id_contrib`. Hard-link the two files into `nn/feat_cache/{split}/`. |
| **FR-2** | **PREP-1 / INVENTORY-1 / RESEARCH-1: prew clip.** Same mechanism for `prew_fe` / `prew_new` (`nn/train.py:40-42, 46`). | [V] by 3 agents. Present rows at 10: 86.1% / 78.7% val, 94% train production. In-model rank AUC 0.562 vs 0.608 unclipped. Standardised spread of the rank channel 1.55x larger on val than in the train stats. The ordering survives in `prew_fe_z` (AUC 0.620). | **yes**: nn_final5 members `v12_allprew_s83/84/85` (in v11) and nn_final6 run `v14_mix1_s91` | Existing checkpoints: none (they were trained on the clipped column). Future cvnd_prew-type members: swap to `prew_fe_s` / `prew_new_s` (one pre-declared choice among `_s` / `_c` / `_wrk`). Low priority: P2 says no more prew A/B runs. |
| **FR-3** | **PREP-2 / SHIFT-1 / LEAKAGE-7 / INTEGRITY-4: `it_hist_*_luo` is non-causal on train rows, and the mismatch sits in young listings.** | [V] Young-tercile zero share: train random-like 0.364 vs val 0.771 / private 0.912. P(pos \| zero)/P(pos) 0.62 vs 0.88. Old tercile matches (0.147 vs 0.187; 0.92 vs 0.91). Largest train(random)→val input shift: SMD +0.33. | all members (inherent) | **None that wins.** Causal −0.0018 ± 0.0008, lagged −0.0027 ± 0.0008 (LEAKAGE-7). it_hist_hz restores parity (zero-lift ratio 0.71 → 0.99) but nets −0.0003..+0.0011 across runs, a cold/warm trade: +0.041 on all-young-positive feeds, −0.006 on the rest (featnew_it_hist_hz §5). Keep as built, and keep the fine-tuned half (it repairs young-cold calibration). |
| **FR-4** | **RESEARCH-2: co-vis censoring mismatch.** | [V] Censored share 8.6% train vs 20.1% val / 31.3% private. Censored-zero lift 0.43 train vs 0.67 val. Train-only NN actual/expected on val cv = 0 & censored 1.055; train-fit GBDT 1.130 ± 0.018 (featnew_hzcv §6). | all members (inherent) | **None that wins.** hzcv fixes the lift (0.43 → 0.70 vs 0.68) but creates a coverage mismatch: 43% of train feeds lose the dense source because the data start on Sep 6. Net +0.00032 ± 0.00071, and calibration is unchanged (1.123). The fine-tune repairs it (0.967). Keep as built. |
| **FR-5** | **INVENTORY-2: the train-fit LightGBM gate base contains parity-broken columns.** The base is `swap_luo(base_features.json)` (`pretrain/gate_pre2.py:21`); it keeps `it_cand_cnt_split*` (train PSI 16.6, sign flip), `it_cand_cnt_train` (self-count) and `hour`. | [V] Removing them: val +0.0029 ± 0.0010, holdout −0.0053 ± 0.0006. pre2 (+0.0054) and prew (+0.0032) gate deltas were measured on this base. | harness only (acceptance numbers) | For any further gate use, a copy with base minus `it_cand_cnt_*` and `hour`; keep `tte_it_imp`. The hz screens already used the clean `lgb_causal` BASE. |
| **FR-6** | **PREP-3 / INVENTORY-4/5: fine-tune extras hygiene. W3 (`n2c_pre_v2`, running, `nn/logs/w3_ftlist.log`) fixes hour, it_tr_\*, u_tr_\*, it_cand_cnt_train, id_contrib_v2 and user-0 singletons, but not the items listed here.** | [V] (re-checked here on val). `pre_idmax32` NaN 55.3% (private 60.1%), mean-imputed with no flag. `pre_hmax32` / `pre_hmax8` −2 sentinel on 1.40% of rows (lands at −6). `seq_cv_norm` duplicates `seq_cv_invd20` (ρ 0.9998), is not logged by the max > 20 rule (max 5.69, skew 24.8). `it_hist_all_luo_z` is not logged (range −1.67..16.7, skew 5.9). 14 of the listed names are missing on disk, so 41 load (W3 log). **Correction to the workflow plan:** `seq_cv_invd20` *is* logged by the rule (val max 93.2 > 20); only `seq_cv_norm` is not. | yes: the fine-tuned half of v11 (n2cpf fta chains) | Fold into the next pre-declared re-FT only: an explicit `pre_idmax32` has-file; drop `seq_cv_norm`; signed log1p of `it_hist_all_luo_z`; prune the phantom names. Expected ≤ +0.0003. Judge with ≥ 2 FT seeds (FT A/A spread is 1.4x SE). |
| **FR-7** | **LEAKAGE-3 / P7: user-0 rows keep later user-0 snapshots in LUO** (`src/features_luo.py:23`, `user_id > 0` only). | [V] Holdout 0.59% of rows, lift 2.5-3.1, vs val 0.036%. | all members (train input; mainly biases the holdout) | Subtract later user-0 occurrences for user-0 rows at the next popularity rebuild, as its own change. |
| **FR-8** | **PREP-4/5: NaN→0 collides with legitimate zeros; no user-vector-missing flag.** | [V] `ctx_hn_top1w` is 0 on 87.7% of val rows. Nearly all NaN rows are whole H = 0 feeds, which `log1pH` identifies. Zero user vectors: 1.42% val vs 0.33% private feeds. | all members | Next from-scratch preset only: `has:ctx_nd_hcent`, `u_missing`, drop `seq_cv_norm`. Expected ~0. |
| **FR-9** | **RESEARCH-3: history-embedding missingness has a split-dependent horizon.** Old history tokens are masked 3-6 pp more often on train. | [V] (this step; `scratchpad/featrev/synth/r3_mask.py`, `r3_eval.log`). The 6 nn_final5 train-only checkpoints were replayed on CPU (corr with stored preds 0.999999), with val non-missing history embeddings masked at the excess rate (q = 0.30 / 1.04 / 3.41 / 2.80% for hpos < 10 / 10-49 / 50-149 / ≥ 150; 2.1% of events), over 2 mask draws. 6-member z-ensemble: draw 0 **−0.00039 ± 0.00022**, draw 1 +0.00003 ± 0.00020, mean −0.00018 ± 0.00017. Single members −0.00074..+0.00020. No split stands out (M<50 −0.00043 / M≥50 −0.00037 on draw 0). | all members | **No action.** Train-like masking at inference does not help val; it only removes a little information. There is no evidence that the horizon difference costs anything, so no train-side augmentation is needed. |
| **FR-10** | **Process: the pre-declared parity detector population is unreliable** (ixh side finding). In p_random > 0.5 with nothing excluded, 68% of feeds have M < 50 (8.4% positives vs val 5.2%), and clean references fail the 0.03 AUC rule: `tte_it_imp` 0.487 vs 0.538. `feat_new/hz_detect.py` uses this population. | [V] featnew_ixh §3, §6 | detectors only | Use p_random > 0.9 & M ≥ 50 against val M ≥ 50, and always run a reference feature through the same rule. |

**Smaller verified notes (no action):**
- `_rk` features rank NaN rows arbitrarily: GBDT only, negligible (INVENTORY-9).
- The `tte_*` unseen-item prior differs in the 9th decimal between train (float32) and val/test (float64) (INVENTORY-7). Only an exact-threshold tree could split on it; GBDT only.
- `ud_gap_l` would extrapolate on test (+0.83 val-sd) if ever used.
- Pretraining content table: train pre2 rows take val/test vectors for 15-21% of shared items, about −0.0002 and not a leak (PREP §2).

---

## 2. Preprocessing recommendations per model

### 2.1 NN train stage (`attach_xt`, `nn/data.py:182-201`; in-model dense features `nn/model.py`)

The current transform:
- `*_rk` raw;
- `*_z` clipped ±5;
- `seq_cv*` / `it_hist*` log1p(max(x, 0));
- everything else clipped ±10;
- `has:` = non-NaN indicator, NaN → 0, fp16;
- then standardisation with the checkpoint's train stats and a ±6 clip (`nn/model.py:419`).

It is correct for every cvnd input [V] (PREP App. A; INVENTORY §5; RESEARCH §2). Rules for new inputs:
1. **Any raw column in XT_RELONLY / XT_REL must stay strictly inside (−10, 10).** Otherwise its rank/z runs on ties (FR-1, FR-2). Use a fixed affine map (x + 20)/6, as in `feat_new/inventory_prew_rescale.py` and `feat_new/pre2_rescale.py`, or feed the precomputed `_z`. **Never patch `attach_xt` in place**: live checkpoints were trained on the clipped columns.
2. **Name by intended transform.** A count must start with `seq_cv` / `it_hist` to get log1p. Never end a non-z column in `_z` / `_rk`.
3. **Give every NaN-bearing input a `has:` flag** when 0 is a legitimate value, or when NaN must not read as a value (`has:ctx_nd_hcent`, `has:ixh_good`-type cases). NaN → 0 must sort below present values; the affine copies guarantee this.
4. **Judge every popularity-, CF- or encoder-type input on train-only val, never on the holdout.** Holdout inflation is 2.5-5x for transductive popularity (LEAKAGE-1; REVIEW_SYNTHESIS §2 narrowed it to popularity counts).
5. **Any train-row time-sensitive input must match val/test parity on M-matched random-like rows before a GPU run.** Use the young-listing table (PREP s14) and the censored-zero lift (featnew_hzcv §3) as standard diagnostics.
6. **No changes are needed for f1..f5** (quantile codes; raw + feed z + feed pct), the ctx_hs sums (clip at 10 costs nothing; AUC identical to 4 decimals), fp16, or the logged counts. PLE / quantile re-encodings were tested null (nnx_VERDICT).

### 2.2 NN fine-tune extra branch (`attach_extra`, `nn/data.py:204-220`; `nn/finetune.py:141-170`)

The current transform: log1p if min ≥ 0 & max > 20 on val; val mean/sd standardisation, reused for test; clip ±6; NaN → val mean; no flags. Recommendations, all for the next pre-declared re-FT after W3:
1. **Explicit has-files for NaN-heavy inputs** (attach_extra has no `has:` support): `pre_idmax32` (55% NaN). Use `ixh_has` / `ud_has`-style files for any new extra.
2. **Map sentinels to NaN plus a flag:** `pre_hmax32` / `pre_hmax8` = −2 on 1.40% of rows.
3. **Signed log1p for unlogged heavy tails:** `it_hist_all_luo_z` (skew 5.9). Drop `seq_cv_norm`, which duplicates `seq_cv_invd20`; that one is already logged.
4. **Keep W3's removals:** hour, it_tr_\*, u_tr_\*, it_cand_cnt_train; `pre_id_contrib_v2` instead of `pre_id_contrib`.
5. **Avoid inputs whose test distribution sits outside val under val-fit stats.** Examples: `ud_gap_l` +0.83 sd; raw staleness counts such as `it_cand_cnt_train` −0.22 sd on private. Prefer within-feed `_rk` / `_z` versions, which stay at z-mean 0.00.
6. **Acceptance:** non-inferiority / > 2 SE with ≥ 2 FT seeds (FT A/A spread 1.4x SE, REVIEW_SYNTHESIS §4).

### 2.3 GBDT harnesses (`src/eval_family.py`, train-fit gates)

- **Raw float32, NaN kept as missing, `category_id` categorical, no scaling.** This is correct, and LightGBM handles the quantile codes natively [V] (PREP §2).
- **Clean the bases before reuse.**
  - Train-fit gate: drop `it_cand_cnt_*` and `hour` (FR-5).
  - `eval_family`: swap_luo and drop `it_cand_cnt_split*`, then clear `data/eval_cache/base_oof_s*` (LEAKAGE-5; about +0.0002 leak in base levels, verdicts unaffected).
- **Prefer NaN over arbitrary ranks for NaN rows in `_rk` columns** (INVENTORY-9). For `tte_*`, compute the prior in float64 on every split (INVENTORY-7).
- **Read GBDT deltas as paired-within-run only.**
  - LightGBM is not bit-exact across thread counts or load: R0 A/A spread 0.0004-0.0011 on identical runs (featnew_it_hist_hz §5.1).
  - Seed 2 of eval_family is a known low base draw (+0.0008 inflation of 3-seed means).
  - Column order changes feature_fraction sampling.

---

## 3. New features: what was built, screens, verdicts, next step

### 3.1 The four planned families (all REJECTED by their adversarial verdicts; no leak flags)

| family | files (`<repo>/data/feats_new/{split}/`) | what it changes | train-fit gate (val / holdout) | residual over NN mix (GBM / LIN) | val-only eval_family | verdict and next step |
|---|---|---|---|---|---|---|
| **hzcv** (`feat_new/hzcv.py`, `hz_common.py`) | `seq_cvlag_{invd20,pop,norm}`, `hzcv_{expo_slog,censored,stale_d}` (all splits), `train/hz_lag` | train-row co-vis from a lagged dense source plus a 1/37 sparse window; censoring offsets. Val/test = byte copies of `seq_cv_*` / `src_expo_*`. | +0.00032 (0.00071) / −0.00057 (0.00041); capped-lag R2 −0.00001 (0.00068) | +0.00023 (0.00041) / −0.00063 (0.00034) | +0.00291 (0.00078), but this re-measures `seq_cv_*` (base lacks `seq_*`); offsets over base + `seq_cv` −0.00064 (0.00065) | **Drop.** Structural: Sep 6 data start → 43% of train feeds lose the dense source (76% at L = 8). Co-vis zero share 0.65 vs 0.42-0.48 on val/test. `hzcv_stale_d` co-varies with "no dense source" on train only. |
| **ixh** (`feat_new/ixh.py`) | `ixh_{good,lpos,mlogM,dlogM,mf1,df1,ln,has}` (all splits) | item exposure context (production-ranker opinion, feed size, f1 audience) from other users' earlier train candidate lists, with the shared lag | R3 +0.00012 (0.00072) / +0.00007 (0.00042); stale-val −0.00109; g1-style +0.00097 (0.00079) / −0.00088 (0.00052) | +0.00053 (0.00047) / −0.00002 (0.00034); on nn_final5 mix −0.00036 / +0.00023 | +0.00064 (0.00069) | **Drop.** At L = 0 its count is exactly `tte_it_imp` (already in bases). It decays with staleness. Under the lag, 64% of train rows are NaN vs 35-47% on val/test. |
| **it_hist_hz** (`feat_new/it_hist_hz.py`) | `it_hist_hz_{all,et1..et4}` (all splits) | age-gated, density-matched LUO for train rows: listings created after the cutoff are counted only in 1/37 train + val/test snapshots. Val/test byte-identical to `it_hist_*_luo`. | primary +0.00111 (0.00075); deterministic 4-seed −0.00027 (0.00058); g1 −0.00067 (0.00076); holdout −0.0010..−0.0015 in every run | −0.00061 (0.00032) (degenerate: val values unchanged) | +0.00077 (0.00068) (degenerate) | **Drop** (keep the files for reference; it answers W7). Parity is restored (zero-lift ratio 0.71 → 0.99; young-row count-to-lift curve matches val), but it is a cold/warm trade: +0.041 (0.0027) on all-young-positive feeds, −0.0053 (0.0008) on the rest. Private-mix reweighting +0.0008 (0.0009-0.0012), n.s. |
| **ud** (`feat_new/ud.py`) | `ud_{cos_prev,drift,drift_fz,gap_l,has}` (val/test only) | user-vector drift since the same user's strictly earlier snapshot (fine-tune extra) | −0.00056 (0.00069) / +0.00034 (0.00042) (train values built in memory) | on nn_final5 mix: +0.00027 (0.00028) / −0.00008 (0.00020) | −0.00039 (0.00067) | **Drop.** The +0.00038 prior came from a base with the pre term; on the nn_final5 mix the same blend is +0.00009 (0.00020). Correction: train previous feeds are 2.9 days old (median), not minutes old. |

All four families pass their construction checks: brute-force or pipeline reproduction [V]. Examples:
- hzcv at L = 0 reproduces `seq_cv_*` exactly (max |diff| 0 on 223,854 rows);
- the ixh brute force matches to ≤ 4.6e-7;
- the it_hist_hz unthinned recount equals LUO on 100% of 1M rows;
- the ud previous-feed lookup has 0/800 mismatches.

None is to receive a GPU slot. Per the workflow rule, **no joint paired NN run (`cvnd_hz`) is warranted**: no family passed its CPU screen.

### 3.2 Built and recommended: clip-safe pre2 (this step) → paired GPU NN run

**Files** (`feat_new/pre2_rescale.py`): `<repo>/data/feats_new/{train,val,test}/pre2_fe_s.npy` and `pre2_new_s.npy`.
- Construction: (x + 20)/6, NaN kept; fixed constants, identical on every split.
- The builder asserts that no value reaches the clip and that the NaN pattern is identical (`scratchpad/featrev/synth/pre2_rescale.log`). Mapped ranges: train [0.715, 8.432], val [1.734, 7.926], test [1.530, 8.009].
- The log reproduces FR-1 on all rows: present rows that `attach_xt` clips today are pre2_fe train 92.46% / val 83.84% / test 84.66%, and pre2_new 87.86% / 75.91% / 76.88%.
- Val per-feed AUC after the fp16 cast (`pre2s_auc.log`): rank channel as fed today **0.5521** (pre2_fe) / 0.5715 (pre2_new), with distinct values per feed 0.19 / 0.26. With `_s`: 0.6632 / 0.6614, equal to the raw columns. `pre2_fe_z` is also 0.6632.
- Within-feed rank equals the unclipped rank (up to fp16 ties: 0.91 distinct per feed, no AUC loss), z is affine in the raw z, and NaN → 0 stays below every present row (minimum mapped value 0.715).

**Preset for the lead** (new name; leave `cvnd_pre2` untouched for its checkpoints):
```python
XT_PRESETS["cvnd_pre2s"] = XT_PRESETS["cvnd"] + ["pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "has:pre2_fe"]
XT_REL["cvnd_pre2s"] = ["seq_cv_invd20", "it_hist_all_luo", "pre2_fe_s", "pre2_new_s", "pre2_hmax32"]
XT_RELONLY["cvnd_pre2s"] = ["pre2_fe_s", "pre2_new_s", "pre2_hmax32"]
```

**Handoff.** `feat_path` reads `nn/feat_cache/<split>/` first, then `data/feats/<split>/` (`nn/data.py:31-42`):
```bash
for s in train val test; do for n in pre2_fe_s pre2_new_s; do ln data/feats_new/$s/$n.npy nn/feat_cache/$s/$n.npy; done; done
```
These are new names, so nothing live changes.

**Run.**
- `--xt cvnd_pre2s`, paired with the W2 baselines `v8_nd_e3` / `v8_nd_e3_s42`: the same 3-epoch args and seeds 23 / 42 as `v16_pre2_e3_*`.
- Report train-only val, holdout, M<50 / M≥50, Tue / Thu and the late slice (`nn/pcmp.py`).

**Acceptance.**
- The unified rule: train-only val > 2 SE over 2 seeds, and holdout ≥ 0.
- Pre-declared as the bug-fixed version of W2, i.e. one choice, not a new search.
- If it passes and nn_final6 gets pre2 members, use `cvnd_pre2s` rather than `cvnd_pre2`.

**Expected.** 0 to +0.0005 over W2's +0.0012 [S]. The full ordering already reaches the model through `pre2_fe_z` / `pre2_new_z` (val AUC 0.6632, identical to raw), so the fix mainly removes the train/val tie-share and id-availability skews.
- Do not spend a dedicated A/B on the fix. W2 is marginal (z 2.0, all of it in M<50), so pre2 needs a confirmation pair anyway, and that pair should use `cvnd_pre2s`. Any pre2 member of nn_final6 should use it too.

### 3.3 Other small builds of the workflow (status)

| files (`data/feats_new/{split}/`) | builder | status / next step |
|---|---|---|
| `prew_fe_s`, `prew_new_s` (all splits) | `feat_new/inventory_prew_rescale.py` | the chosen fix for FR-2; only for a future cvnd_prew-type member (low priority) |
| `prew_fe_c`, `prew_new_c` (NaN → −10); `prew_{fe,new,hmax32}_wrk` (warm-only rank) | `feat_new/research_expo.py`, `feat_new/prew_wrk.py` | alternatives to `_s`; do not test all three (≤ 3 choices) → drop |
| `src_expo_h`, `src_expo_slog`, `src_censored` (all splits) | `feat_new/research_expo.py` | nested blend +0.00028 (0.00041); offsets add −0.0006 over base + seq_cv and +0.0002 over the NN → drop |
| `ctx_nd_{c80,c90,c95,mean,dupb_h,dupb_f1,dupb_u}` (val/test) | `feat_new/inventory_ctx_nd_restore.py` (bit-exact) | residual +0.0001 / +0.0005 (0.0004) over the mix, ~0 over train-only → optional n2full reproducibility only |

---

## 4. Remaining ideas (ranked by expected value per cost) and things verified correct

### 4.1 Ranked

| rank | idea | exp. gain (private) | cost | note |
|---|---|---|---|---|
| 1 | **`cvnd_pre2s` as the W2 confirmation pair** (section 3.2); if it passes, use it for nn_final6 pre2 members | pre2 itself +0.0012 (W2, marginal); the fix 0 to +0.0005 on top | 2 × 3-epoch GPU runs (~45 min each) vs the existing v8_nd_e3 / _s42 baselines | the only feature family with a ≥ 2 SE train-stage signal |
| 2 | **Fine-tune complements in the next re-FT** (FR-6: `pre_idmax32` has-file, −2 sentinel → NaN + flag, drop `seq_cv_norm`, signed log1p of `it_hist_all_luo_z`, prune phantom names) | ≤ +0.0003 | < 1 GPU-h per member set, ≥ 2 FT seeds | no extra val choice if bundled with a planned re-FT |
| 3 | **Clean gate / eval_family bases** (FR-5, LEAKAGE-5) | decision quality | CPU minutes | before any further GBDT screen |
| 4 | **LUO user-0 rule** (FR-7) at the next popularity rebuild | ≤ 0.0005, mostly holdout honesty | CPU ~10 min + rides on a run | its own paired change |
| 5 | **Regime flags in the next from-scratch preset** (`has:ctx_nd_hcent`, `u_missing`, drop `seq_cv_norm`) | ~0 | free when a preset changes anyway | hygiene |
| 6 | **prew `_s` swap** for any further prew member (FR-2) | 0 to +0.0005 | free with a planned member | P2: no dedicated A/B |
| 7 | **Gap-matched `tte_*` for a train-stage CTR prior** (INVENTORY-7 idea 3) | ≤ +0.0005 | CPU 30 min + a run | residual over train-only only +0.0001..+0.0004; low |

**Do not re-propose** (measured null or structurally blocked):
- hzcv, ixh, it_hist_hz, ud;
- causal or lagged it_hist; W7 (×37 weighting, superseded by it_hist_hz);
- PREP's plain thinning; `src_expo` standalone;
- time-windowed ALS, T1 / T2 taste profiles, life_ / ua_ freshness, f5 trend;
- listing-edit flags, re-exposure, in-split candidate counts;
- hour / calendar; GBDT leaf embeddings;
- log copies of ctx_hs; restoring pruned ctx_nd;
- val-like history truncation; more prew A/B runs;
- the workflow plan's rejected list.

### 4.2 Verified correct (safe foundations)

- **Alignment and integrity.** All feature files have the split's candidate count (256 val / 256 test / 238 train + the new ones). Histories contain no post-request events, and no candidate is in its own history (REVIEW_SYNTHESIS §3).
- **Train-stage inputs are label-free.**
  - `seq_cv_*` is causal: train rows use a day-cutoff source with same-user sequences skipped. It is also aligned across splits in zero rate and conditional lift (SHIFT §2).
  - `ctx_*` are feed-internal.
  - LUO arithmetic equals a raw recomputation exactly, and same-user-later removal targets exactly the leaking rows (91-93% positive).
  - The item-ID path is off in all nn_final4/5 members (`id_buckets 0`).
- **Organiser features.**
  - f1..f5 show no in-sample inflation on train (sAUC within 0.004).
  - They are stable over all train / val / test days (≤ 0.13 train sd).
  - f4 = exact listing age; f3 is constant within (feed, category).
- **Embeddings.**
  - Each split's own item table is the most time-local (swaps cost 0.0002-0.0005 AUC, so later tables encode no later interactions).
  - The user embedding is a per-request vector with no future information (flat AUC over train days; own-feed vector lacks own clicks: cos 0.363 vs 0.394 for the next feed).
  - History tokens without an embedding are masked, not pooled as zeros.
- **NN scaling.**
  - 96 of 104 inputs have val z-sd within 0.75-1.25, and the 8 others are explained.
  - In-model rank/z channels are split-invariant (z-mean 0.00, sd ≈ 1).
  - fp16 is lossless at 4-decimal AUC.
  - Val→private drift of every train-stage input is ≤ 0.16 SMD.
  - Normalisation stats are an identical affine map on every split.
- **Fine-tune.**
  - GroupKFold by user; label-free val-fit standardisation reused for test.
  - No ALS, x_lh, non-LUO it_hist or it_cand_cnt_split among the extras.
  - W3 now uses `pre_id_contrib_v2` and user-0 singleton groups (W3 log).
- **Parity of the new families.** All four have identical missingness mechanisms on val and test. Val/test inputs of hzcv and it_hist_hz are byte-identical to the audited as-built files. The hz holdout is forced to L = 0 and matches `common_rows.holdout_feeds()` exactly.
- **Guardrail P6** is closed (nn_report l.488; `nn/train.py:37`; no `pre_*` in `nn/feat_cache/train/`).
