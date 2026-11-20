# R7: modern architecture blocks and never-varied hyperparameters. Decision report

Written 2026-11-12 08:15 UTC by the r7 reporting step. Details: `r7_research.md` (selection, priors, specs), `r7_build.md` (code and 321/321 GPU tests),
`r7_screen.md` (pre-declaration frozen before launch, full readouts). Code is in `nn_x/r7/`. Every arm is a flag whose default reproduces nn/ bit for bit.

## 0. Decision

- **Nothing to adopt.** 0 of 13 arms passes the pre-declared stage-1 rule, so there was no stage 2 and no stage 3. Every r7 flag stays at its OFF default.
  The next families keep the current block and training recipe unchanged:
  - pre-LN LayerNorm blocks, GELU FFN ×2, d = 128, 4 heads, 2 blocks, 1 history layer;
  - dropout 0.1 at every site;
  - Muon on every 2-D matrix, momentum 0.95; AdamW betas (0.9, 0.98);
  - batch 32,768 candidates / 512 feeds, warmup 500.
- **The modern-block question is answered for this model, at 2-epoch screen power** (smallest detectable effect ≈ +0.0008 val).
  - No norm, residual scheme, FFN form, attention form or tabular input encoder makes a stronger 2-epoch member.
  - None makes a usefully more diverse one: none clears the diversity margin.
  - Derf (norm-free), Peri-LN and batch ×2 (square-root rule) are clearly harmful on both val and holdout.
- **The only structural lift found this morning is r6's, not r7's.** r6 a4u (retrieval slots made unique, with per-type counts) passes:
  - val +0.00117 (z +2.48) on r6's baseline, and val +0.00102 (z +2.01) on r7's multi-replicate baseline;
  - it is the only change that does not cost M≥50 feeds.

  r6's 10-epoch pairs for a4u and cset decide it (§6). Information access, through retrieval, moves the model; block form does not.
- **The never-varied hyperparameters have now all been analysed** (table in §5). Those that a 2-epoch run can see were measured: batch, optimiser partition, history depth, and dropout at 2 epochs.
  - All are null or worse, so the current values stay.
  - What remains is horizon-only: dropout in the cooldown, cooldown length, and the aux-anneal target. These need the 10-epoch cooldown regime (continuation harness, §6).

## 1. What was tested and why

**Why:** the ensemble is saturated. The pool gain is about +0.0009 val over v12, and the cross-family correlation is 0.92-0.94 even for GRU vs transformer, so stronger single members are the lever.
The user asked for modern blocks (tabular-DL advances, norms, residual schemes, hyper-connections) and for a proper selection of the never-varied settings.

| group | arm (flag) | mechanism | prior (research) | cost / step | params |
|---|---|---|---|---|---|
| optimiser partition | muon_in (`--muon_inputs adamw`) | 5 input projections on AdamW (per-coordinate steps), not Muon | −0.0003..+0.0010 | 1.00 | 0 |
| batch | batch2s (`--cand_budget 65536 --max_feeds 1024 --lr 1.414e-3 --warmup 250 --eval_every 500 --muon_momentum 0.90 --adam_betas 0.8,0.96`) | batch ×2 under the pre-declared SDE **square-root** rule; meant as a cost lever | non-inferior, ≤ 0.75× time | 0.82 per sample | 0 |
| dropout | resdrop0 (`--res_drop 0`) | residual / FFN / retrieval dropout 0.1 → 0; attention-probability dropout stays at 0.1 | +0.0002..+0.0010 at 2 ep | 0.99 | 0 |
| tabular encoder | pbld (`--pbld 16`) | RealMLP PBLD periodic embeddings, zero-init into c_mlp | 0..+0.0008 | 1.10 | +88.5K |
| tabular encoder | ftok (`--ftok 8`) | FT-Transformer-lite: 8 feature-group tokens + CLS through 1 block | 0..+0.0006 | 1.47 | +181K |
| residual scheme | hc4 (`--hc 4`) | dynamic hyper-connections, 4 streams over the 8 candidate sublayers | 0..+0.0006 | 1.43 | +7.4K |
| residual scheme | dwa (`--dwa`) | DenseFormer depth-weighted averaging (14 scalars, identity init) | 0..+0.0005 | 1.04 | +14 |
| norm | derf (`--norm derf`) | norm-free erf point-wise (successor to DyT) in the blocks and heads | −0.0010..+0.0008 | 1.06 | +28 |
| norm placement | periln (`--periln`) | Peri-LN output norms on every sublayer | −0.0005..+0.0005 | 1.06 | +2.6K |
| norm placement | postln (`--postln`, reserve) | Post-LN + DeepNorm α / β init | −0.002..+0.0005 | 1.00 | 0 |
| FFN | swiglu (`--ffn swiglu`) | SwiGLU at matched parameters | 0..+0.0005 | 1.05 | −860 |
| attention | diff2 (`--diff_attn cross`) | DIFF-V2 differential target attention in the 2 cross blocks | −0.0005..+0.0008 | 1.11 | +34K |
| depth | hl2 (`--hist_extra 1`) | second history layer, zero-init outputs | 0..+0.0005 | 1.15 | +149K |

Considered and not screened (reasons in r7_research §7):
- RMSNorm: same function class at d = 128.
- DyT: dominated by Derf.
- nGPT: conflicts with Muon and decoupled wd.
- mHC: its doubly-stochastic mixing forbids the amplification the model uses.
- LayerScale / ReZero / SkipInit: covered by DWA and by hl2's zero-init.
- Value residual, GeGLU.
- Heads 8 / blocks 3: closed earlier.
- ff ratio: the FFN is the least-used sublayer.
- Warmup, AdamW betas, clip, Muon momentum alone: no instability to fix, and momentum is a de-noising knob.
- Robust scaling: no headroom.
- TabM / PLE / MoE: closed.

The r6-owned arms (a4u, cset, typed, adrop0, a4w; wd in r6 Q1) were not duplicated.

## 2. Method (pre-declared, frozen 05:23:58 before the first launch at 05:23:59)

- Stage 1: a 2-epoch screen, seeds 23 and 24.
  - It is paired per feed against `r7_base_s23/s24`, re-run in the same code copy as the first launches of the same batch.
  - An A/A same-seed rerun (`r7_base_s23r`) checks the copy.
- Metric: per-feed NDCG of the train-only predictions at the holdout-selected checkpoint.
  - Val: 14,423 feeds. Holdout: 30K M≥50 random-like train feeds.
  - SE_cal = sqrt(SE_feed² + σ_run²/2), with σ_run = 0.00035 on val and 0.00015 on the holdout.
- PROMOTE iff all of:
  1. pooled val Δ > 1.5 SE_cal;
  2. pooled holdout Δ ≥ 0;
  3. both seeds positive on val, or both on the holdout.
- batch2s also has a COST-PASS check: val ≥ −0.0005, holdout ≥ −0.0003, and time ≤ 0.75× the paired baseline.
- Stage 2 (10-epoch pairs, unified rule: val > 2 SE, holdout ≥ 0, no slice < −2 SE) and stage 3 (ensemble z-mean and correlation) apply only to promoted arms.
- Lr rule: square-root for the batch arm. No other arm changes the lr or the batch.

## 3. Stage-1 results: 0 / 13 promote (29/29 runs, 0 crashes, 05:24-07:46 UTC, 16.7 lane-hours)

- A/A: val +0.00003, holdout −0.00006. The copy is sound and σ_run is kept.
- Baselines: s23 val 0.39408 / holdout 0.31779; s24 val 0.39334 / holdout 0.31749.
- Pred-private Δ is the val Δ re-weighted to the private feed-size mix (`src/calib.py`). It is information only and changes no sign and no verdict.

| arm | val Δ s23 / s24 | val pooled Δ (z) | hold Δ s23 / s24 | hold pooled Δ (z) | pred-private Δ (z) | worst slice (z) | verdict |
|---|---|---|---|---|---|---|---|
| dwa | +0.00013 / +0.00114 | +0.00064 (+1.32) | −0.00040 / −0.00014 | −0.00027 (−1.07) | +0.00057 (+1.21) | M≥50 −0.85 | FAIL (1, 2): nearest miss |
| resdrop0 | −0.00030 / +0.00091 | +0.00030 (+0.53) | +0.00038 / +0.00042 | +0.00040 (+1.29) | +0.00024 (+0.43) | M≥50 −0.65 | FAIL (1) |
| hc4 | −0.00032 / −0.00013 | −0.00022 (−0.42) | +0.00035 / −0.00018 | +0.00008 (+0.30) | −0.00026 (−0.49) | M≥50 −2.30 | FAIL (1, 3) |
| postln | −0.00130 / +0.00041 | −0.00045 (−0.82) | −0.00036 / +0.00023 | −0.00007 (−0.22) | −0.00048 (−0.88) | Thu −2.18 | FAIL (1, 2, 3) |
| pbld | −0.00060 / −0.00054 | −0.00057 (−0.93) | +0.00026 / +0.00058 | +0.00042 (+1.22) | −0.00057 (−0.95) | Thu −2.22 | FAIL (1) |
| hl2 | −0.00020 / −0.00098 | −0.00059 (−1.11) | +0.00007 / +0.00007 | +0.00007 (+0.26) | −0.00066 (−1.25) | M≥50 −3.27 | FAIL (1) |
| diff2 | −0.00141 / −0.00046 | −0.00093 (−1.73) | −0.00006 / −0.00023 | −0.00014 (−0.49) | −0.00097 (−1.82) | Thu −3.19 | FAIL (1, 2, 3) |
| muon_in | −0.00049 / −0.00154 | −0.00101 (−1.79) | −0.00001 / −0.00060 | −0.00031 (−0.98) | −0.00107 (−1.92) | M≥50 −2.51 | FAIL (1, 2, 3) |
| swiglu | −0.00159 / −0.00054 | −0.00107 (−1.95) | +0.00013 / +0.00024 | +0.00019 (+0.60) | −0.00114 (−2.11) | M≥50 −2.54 | FAIL (1) |
| periln | −0.00110 / −0.00111 | −0.00111 (−1.97) | −0.00083 / −0.00044 | **−0.00064 (−2.02)** | −0.00114 (−2.06) | M≥50 −2.91 | FAIL; harmful |
| ftok | −0.00124 / −0.00104 | **−0.00114 (−2.12)** | −0.00025 / +0.00012 | −0.00007 (−0.22) | −0.00119 (−2.24) | Thu −3.42 | FAIL; val-harmful |
| derf | −0.00116 / −0.00132 | **−0.00124 (−2.15)** | −0.00069 / −0.00122 | **−0.00096 (−2.91)** | −0.00132 (−2.31) | M≥50 −3.67 | FAIL; harmful |
| batch2s | −0.00164 / −0.00126 | **−0.00145 (−2.47)** | −0.00071 / −0.00148 | **−0.00109 (−3.44)** | −0.00149 (−2.59) | M≥50 −4.17 | FAIL; COST FAIL (0.815× time > 0.75×) |

Conditions: 1 = pooled val Δ > 1.5 SE_cal; 2 = pooled holdout Δ ≥ 0; 3 = both seeds positive on val or on the holdout. SE_cal is 0.00048-0.00061 on val and 0.00025-0.00034 on the holdout.

### Patterns (information only, not part of the rule)

- **Robust to the baseline draw.** Both r7 baselines landed high among the same-seed replicates (sx, r6 and r7 copies). Re-scored against the replicate mean, 0 / 13 still promote:
  - dwa: val +0.00088 (z +1.92), but holdout −0.00025 in both seeds;
  - resdrop0: val +0.00055, holdout +0.00042.
- **Every perturbation costs warm / M≥50 feeds.** Val M≥50 is negative for 13/13 arms (mean −0.0013). The small gains sit in M<50 and cold feeds:
  - dwa: M<50 +0.0021 (z +2.6);
  - pbld: coldpos +0.0033 (z +2.8);
  - hc4, resdrop0, hl2: M<50 +0.0011 to +0.0013.

  This is post hoc (13 arms × 6 slices), and the private feed-size re-weighting does not change it. It is not evidence for promotion.
- **Diversity.** Wording corrected from r7_screen §3.4. 11/13 arms have ρ_x slightly below the seed-change ρ_bb = 0.9781; the exceptions are batch2s 0.9789 and hl2 0.9784. No arm reaches the pre-declared margin ρ_x ≤ ρ_bb − 0.005 = 0.9731; the lowest is resdrop0 at 0.9745. So no arm is a same-strength diversity member for the ensemble.

## 4. Stage 2, stage 3 and verification

- **Stage 2:** nothing to run, because 0 arms promoted. The pre-declared drop0 combination is not triggered, since neither resdrop0 nor r6 adrop0 promoted.
- **Stage 3:** not applicable. Information only: the 2-arm vs 2-base-member z-mean is ≥ +2 SE on no arm. Best case is dwa, val +0.00091 (z +1.7) with holdout −0.00045.
- **Verification: CONFIRMED.**
  - Every treatment run differs from its paired baseline only by the arm flags. batch2s differs by exactly the 7 pre-declared values.
  - The flag-off code keeps the nn/ path verbatim: 321/321 tests, including bit-identical logits, loss, gradients and RNG.
  - Run diagnostics show every arm was active, for example pbld P 0 → 22.5, ftok Zf 0 → 7.1, hl2 W_O 0 → 10.3, and "muon: 41 matrices" for muon_in.
  - The code md5s equal the frozen ones.
  - An independent NDCG recompute reproduces every Δ, SE, z and slice, and the A/A.
  - No leakage path: the new modules are per-candidate or reuse the existing key masks, and preds come from the holdout-selected checkpoint.
  - GPU rules were respected, and no other job was touched.
- Minor issues, none of which changes a verdict:
  - The diversity wording, corrected above.
  - Three interim arm reads preceded the A/A step that §1.4 said comes first. The A/A passed, σ was left as declared, and the verdicts hold for any σ ≥ 0.
  - The header time "frozen at 05:25" should read 05:23:58.

## 5. Never-varied hyperparameters: status after r7 and r6

| knob (current) | how it was selected | result | status |
|---|---|---|---|
| batch 32,768 / 512 feeds | r7 batch2s, square-root rule | val −0.00145, holdout −0.00109. Throughput 0.815× per sample, so it saves only 18% | **keep**. Closed as a cost lever: throughput fails the 0.75× bar whatever the lr. The linear rule (lr 2e-3) is untested and is an lr question, not a batch question: under Muon, distance ∝ steps × lr, which is 0.71× under the square-root rule and 1.0× under the linear rule |
| optimiser partition (Muon on input projections) | r7 muon_in | val −0.00101, holdout −0.00031 | **keep Muon on all 2-D matrices** |
| residual / FFN / retrieval dropout 0.1 | r7 resdrop0 (2 ep) | +0.00030 / +0.00040 (holdout positive in both seeds): null at 2 ep | **keep 0.1 for now**. The 10-epoch cooldown effect is open (§6) |
| attention dropout 0.1 | r6 adrop0 | FAIL | keep |
| hist_layers 1 | r7 hl2 | val −0.00059, holdout +0.00007 | **keep 1** |
| norm (pre-LN LayerNorm) | r7 derf / periln / postln | −0.0012 / −0.0011 / −0.0005 val | **keep pre-LN LN** |
| FFN form (GELU ×2) | r7 swiglu | val −0.00107 | keep |
| residual (plain pre-LN) | r7 hc4 / dwa | null / nearest miss with negative holdout | keep |
| heads 4 / blocks 2 / width 128 | earlier: heads 8 −0.0011, blocks 3 −0.0006, d192 weaker (F12, R6_PLAN) | - | keep |
| retr_m 16 / retrieval form | r6 a4w FAIL; **r6 a4u PASS** | a4u +0.00117 val | **r6 10-epoch pair decides** |
| wd 0.02 | r6 Q1 w10 (cooldown continuation) | pending (r104 running) | r6 |
| cooldown pool mix 0.1 | r6 Q1 m02 | pending | r6 |
| cooldown_frac, aux-anneal target, cooldown dropout | horizon-only; 2-epoch screens cannot see them | - | **continuation harness (§6)** |
| warmup 500, AdamW betas, clip, Muon momentum alone | analysed in r7_research §4.1: no instability; momentum is a de-noising knob, which hurts the pool | - | deliberately not screened; keep |

## 6. Adopt / closed / next steps

**Adopt:** none. Exact flags for the next families: no r7 flag. Every new flag stays at its default (`--muon_inputs muon --res_drop -1 --pbld 0 --hc 0 --ffn gelu --norm ln --hist_extra 0 --ftok 0`, no `--dwa`, `--diff_attn`, `--periln` or `--postln`).

**Closed** (at 2-epoch screen power, ≈ +0.0008 val detectable; reopen only with a new mechanism):
- harmful: derf, periln, batch2s (square-root rule);
- null: muon_in, pbld, hc4, dwa, diff2, swiglu, hl2, ftok, postln;
- resdrop0 is null at 2 epochs only.

**Next steps, in priority order** (suggestions for the lead and r6; nothing was launched by this step):
1. **r6 a4u and cset 10-epoch pairs** (r6-owned): `bash nn_x/r6/queue_morning.sh e10 a4u "<2 GPUs>"`, and the same for cset.
   - a4u is the one structural change this morning that lifts val in both seeds without costing M≥50.
   - GPUs N and 3 were idle at 08:09 UTC.
   - Judge it at ensemble level as well: the plateau-averaging lesson applies.
2. **Horizon knobs in the continuation harness** (r6 wave 2, from the F7 @77500 snapshots, same 4-root rule):
   - add a **d0** arm (all dropout 0 after `--init`) next to the planned d20. This brackets dropout in the memorising cooldown, which is where resdrop0's mechanism would act.
   - Then cooldown length {0.5, 1, 2} epochs and aux_anneal {0.1, 0.5} (r7_research §4.2).
   - Note: nn_x/r7/train.py refuses model flags with `--init`. The dropout override must set the module p after loading. Dropout has no parameters, so this is exact.
3. **Stop block-form work for this model.** 9 block arms, 0 promotions, every one costing M≥50.
   - Two more ideas fit the data but are not recommended: dwa as a 10-epoch pair (~4.5 GPU-h; P(≥ +0.0015) ≈ 0.1; negative holdout), and a cold-weighted selection criterion for the cold-tilted arms (post hoc; it would need its own pre-declaration).
4. **Not recommended:** batch ×2 under the linear rule. The cost bar fails on throughput alone, and a strength gain would be the lr effect already measured between F7 (lr 2e-3) and F9 (lr 1e-3).

## 7. Workflow status (08:10 UTC)

- **r7:** research → build → stage-1 screen: DONE. No r7 process is running (RUNNING.txt has END lines for all 29 runs and the driver).
- **r6:**
  - Q2 structure screen: DONE 07:59. a4u PASS and cset FAMILY-CANDIDATE go to 10-epoch pairs, which have not been launched yet. typed, adrop0 and a4w FAIL.
  - Q1 cooldown continuations (GPU N): c0 / m02 / w10 done for roots 101-103 and c0 for 104; m02 / w10 r104 are running (~20 min in). The readout follows automatically.
- **Lead families:** F10 s505/s506 (train-only and FT preds 07:12-07:20) and all 7 F12 d192 seeds have preds, so the pre-declared overnight family decision is unblocked.
- **LLM large matched test** (GPUs N). Ranker baselines on the matched 150K-feed split score holdout 0.30448 (B23) and 0.30325 (B24). The LLM arms trail them by about 0.004-0.006:
  - Qwen3-0.6B: done, best 0.29944 (s23) and 0.29921 (s24);
  - Qwen3.5-4B: best so far 0.29857 (s23, now at step 1200/1514) and 0.29765 (s24, step 1000/1512);
  - Qwen3.5-9B: 0.29790 (s23, step 800) and 0.29409 (s24, step 400). Still running.

Files:
- `nn_x/r7/logs/eval_screen.{txt,json}` and `eval_aa.{txt,json}`;
- `nn_x/r7/logs/r7_screen_predeclaration_frozen.md`;
- `nn_x/r7/preds/` (val and holdout preds of all 29 runs);
- pred-private readout: `/tmp/scratchpad/r7/report/privw.{py,json}`.
