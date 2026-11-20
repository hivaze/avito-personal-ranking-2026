# R6 recap: coverage map and meta-analysis of every training run

Written 2026-11-12 ~01:00-02:00 UTC by the r6 recap agent. CPU only (`CUDA_VISIBLE_DEVICES=""`, <= 12 threads, <= 2 heavy
processes). Nothing under nn/, src/, pretrain/, data/feats/, preds/, subs/ or nn_x/solvers/ was modified.

Inputs:
- every run log: nn/logs/*.log, nn_x/{attn,feat,solvers}/logs, nnx/optim/logs (+aborted), nn_x/verify/*/logs
  (nn_x/optim is a symlink to nnx/optim and was not double-counted);
- the stored train-only predictions in preds/ and preds/holdout/;
- reports CURRENT_STATUS (3i-3q), SOLUTIONS_REVIEW, ROUND5, nnx_VERDICT, REVIEW_SYNTHESIS, FEATURES_REVIEW, sol_*, nn_report.

Code and outputs (all re-runnable on CPU):
- nn_x/r6/recap/parse_runs.py -> runs.csv (one row per train.py segment), ft_runs.csv, curves.jsonl (per-eval curves);
- nn_x/r6/recap/curves.py -> curves_report.txt (family curves);
- nn_x/r6/recap/paired.py -> paired_report.txt (matched comparisons);
- nn_x/r6/recap/groups.py -> groups.csv (family table);
- nn_x/r6/recap_gns.py (gradient-noise scale), nn_x/r6/recap_swa.py and nn_x/r6/recap_avg7.py (plateau weight averaging),
  logs in nn_x/r6/logs/;
- nn_x/r6/recap/avg7_eval.py (7-seed evaluation of the averaging).

Notation:
- TO = train-only.
- "val" = full-list NDCG on the 14,423 val feeds.
- "holdout" = the fixed 30K random-like M>=50 train holdout.
- Families: F5 = v12_* (lr 2e-3, mix 0.3); F6 = v14_s92-96 + v13_lr1_mix1 (lr 1e-3, mix 0.1, cvnd); F7 = v17_pre2s_s101-107 (lr 2e-3, mix 0.1, pre2s);
  F8 = v20_pre2s_e2_s301-312 (2 epochs); F9 = v22_pre2s_lr1_s401-406 (lr 1e-3, mix 0.1, pre2s); F5cd / F6cd = 1-epoch cooldown continuations.
- All Muon+WSD families: cooldown_frac 0.2, retr16 + gate, K9 aux tau 10 / 0.5 annealed to 0.1, d 128 / 4 heads / 2 blocks / 1 history layer.

---------------------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **Inventory.** 512 log files hold 285 train.py segments (277 after removing relaunch copies, 229 complete) and 247 fine-tune segments.
   102 segments use the current optimiser (Muon + WSD).
2. **Never varied in any of the 285 segments:** dropout 0.1, weight decay 0.02, cand_budget 32768 / max_feeds 512
   (~317 feeds and ~25K real candidates per step), max_hist 256 (the data cap), ff 2.
   Never varied under Muon + WSD:
   - blocks 2, hist_layers 1, heads 4 (F12 d192/6h is still queued);
   - retr_m 16 (only 16 vs 0 was ever run);
   - cooldown_frac 0.2 (apart from the 1-epoch continuations), aux_anneal 0.1, warmup 500, lr floor 3%;
   - Muon momentum 0.95 (Nesterov), AdamW betas (0.9, 0.98), global grad clip 1.0.
   Fine-tune knobs that were never varied: lr 5e-5 (231/247 runs; the other 16 are pre-fta AdamW runs at 1e-4/1e-3), wd 0.02, batch 8192/128, EMA 0.99 / start 0.5, extra_lr_mult 10.
3. **Shape of the 10-epoch curves (new synthesis).**
   - **The plateau is noise-dominated.**
     - Eval-to-eval val jitter around the trend is sd 0.0007-0.0009, the same size as the seed spread.
     - The gradient-noise scale at plateau-end checkpoints is B_simple = 4,900 +- 800 feeds (F9, lr 1e-3) and 9,400 +- 3,100 feeds (F7, lr 2e-3), against a batch of ~317 feeds (24,576 feeds per checkpoint, jackknife SE).
   - **The late cooldown over-fits in 19/21 runs at mix 0.1.**
     - End minus selected step: val -0.0033, holdout -0.0041.
     - The effect depends on the dose of pool replay: mix 0 gives -0.0007 / -0.0009 (1 run), mix 0.1 gives -0.0033 / -0.0041, mix 0.3 gives -0.027 / -0.028.
     - At mix 0.1 it is half as large with lr 2e-3 (-0.0020 val) as with lr 1e-3 (-0.0043).
   - **2-epoch cooldowns never decline:** 29/29 runs select within the last two evals.
   - **Where the gain arrives.**
     - Val rises +0.0025 during the constant-lr plateau (20k -> 77.5k).
     - In the cooldown the best point comes while the lr is ~0.5-1.5e-3: 1-9k steps into the cooldown at peak lr 2e-3 (lr 1.5e-3 -> 7e-4 at 80-87.5k), and at the plateau end or the first 6k cooldown steps at peak lr 1e-3. After that the lr goes lower and the model memorises.
   - **At ensemble level (new, CPU):** the 7-seed F7 ensemble is the same or worse at the plateau end as at 60k (0.39551 vs 0.39660, -0.0011 +- 0.0006). The early cooldown adds **+0.0031 +- 0.0006** to it (0.39866). The plateau's last ~1.8 epochs buy nothing for the pool; the cooldown through lr ~1e-3 does.
4. **NEW, measured here, pre-declared: plateau weight averaging improves single members but destroys ensemble diversity. Closed for the pool.**
   - avg3 = the uniform mean of the three saved plateau snapshots of a run (60k / 70k / 77.5k, constant lr). Compared with the same run's holdout-selected cooled checkpoint, all 7 F7 seeds, same CPU fp32 pipeline:
     - single member: **+0.00041 +- 0.00040 val (z 1.0) / +0.00072 holdout**, 6/7 seeds positive. This is below the pre-declared 2-SE bar; the s101 probe (+0.0024) was the extreme seed.
     - averaged members are near-duplicates across seeds: within-feed seed correlation **0.975 vs 0.941** for the cooled members.
     - their 7-member ensemble is **-0.0025 val / -0.0016 holdout** below the cooled 7-member ensemble.
   - **Conclusion: the plateau iterate noise that averaging removes IS the cross-seed diversity the ensemble feeds on.** Section 5.
5. **Ensemble diversity is a recipe property, not just a seed property.** Within-feed seed correlation and 4-seed TO ensemble gain:

   | recipe | seed correlation | 4-seed ensemble gain (val) |
   |---|---|---|
   | AdamW cosine 10 ep | 0.967 | +0.0015 |
   | Muon/WSD 10 ep (F6/F7/F9) | 0.930-0.940 | +0.0037-0.0042 |
   | Muon/WSD 2 ep (F8) | 0.978 | +0.0004 |

   The long noisy constant-lr plateau is what generates member diversity. A change that shortens or de-noises the plateau must be judged at ensemble level.
6. **Matched effects** (single member, TO). z_run = run-level z from the seed spread.

   | comparison | val | holdout |
   |---|---|---|
   | lr 2e-3 vs 1e-3 at 10 ep (F7 vs F9) | +0.0008 (z_run 1.4) | +0.0007 (z_run 2.9) |
   | 10 vs 2 epochs (F9 vs F8), single member | +0.0006 | +0.0007 (z_run 4.0) |
   | 10 vs 2 epochs, 6-seed ensemble | **+0.0040** | +0.0031 |
   | pre2s vs cvnd at 10 ep (F9 vs F6) | +0.0009 | +0.0005 |
   | 1-epoch cooldown continuation (F5cd vs F5) | +0.0018 (z 3.5) | +0.0015 |

   Architecture: GRU ≈ tf, deep ≈ tf, d256 ≈ 0, heads 8 -0.001 (old recipe), DIN -0.0012 to -0.004 in all three recipes. The live F11 DIN is at -0.0018 val / -0.0013 holdout vs the F7 mean at the same steps.
7. **GPU non-determinism.**
   - Same-seed, same-config 10-epoch runs (v13 trio up to 77.5k) diverge to |Δval| 0.0007 / |Δhold| 0.0003 in the late plateau.
   - That is ~65% of the different-seed spread (0.0010 / 0.0005).
   - So same-seed pairing helps little at 10 epochs. It helps more at 2 epochs and in short continuations (|Δ| 0.0002-0.0005 in the first 10k steps).
8. **Ranked proposals (section 7). Judge every one at ensemble level.**
   - **P1** earlier cooldown as a cost lever. Test: 1-epoch cooldown continuations from the existing F7 @60000 snapshots (7 seeds, ~2 GPU-h total). Accept if the 7-seed ensemble is within 0.0005 of F7. If so, new families run at ~7 epochs (~30% cheaper).
   - **P2** cooldown-phase regularisation screen run as continuations from shared @77.5k snapshots: wd 0.1, dropout 0.2, no aux anneal, P8 pool-aux fix, mix 0, history-event dropout, larger pool.
   - **P3** batch x2 at constant noise temperature (lr x2) as a throughput lever. B_simple supports it; it must be confirmed at ensemble level.
   - **P4** retr_m 8 / 32 in the 2-epoch harness (low-noise, mean-function probe).
   - **Closed here:** weight averaging as members, a longer plateau, lr > 2e-3.
   - **Low:** capacity knobs, warmup, optimiser betas, fine-tune knobs.

---------------------------------------------------------------------------------------------------------------------

## 1. Run inventory

Parser: `nn_x/r6/recap/parse_runs.py`.
- Each `==== <time> args: {...}` line opens a segment (Logger appends, so killed-and-relaunched runs give 2 segments per log).
- From each segment it reads the eval lines `step s/T loss val_ndcg holdout_ndcg lr [cd_loss_reg cd_loss_pool]`, the `BEST step` line and `params`.
- Fine-tune segments give the OOF-by-epoch curve, the snapshot mean and the EMA OOF.
- The full per-run table (variant, epochs, lr, optimiser, schedule, cooldown_frac / mix, retr_m, gate, arch, d / heads / blocks / hist_layers / ff / drop / wd / warmup / batch, xt, seed, selected step, val / holdout at the selected step, max val and its step, last val / holdout, non-default extras) is in `nn_x/r6/recap/runs.csv`.
  - 285 rows, 64 columns.
  - Curves are in `curves.jsonl`.

Family table (`groups.py`). val / hold are taken at the holdout-selected step of complete runs. sel_frac = selected step / total steps (median [range]).

| group | segs (complete) | epochs | lr | opt / sched | mix | arch, d/heads | xt | TO val (mean +- sd) | TO holdout | sel_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| round-1 AdamW 2ep (base/nopos/aux/sim/heads 8/blocks 3/d256) | 21 (18) | 2 | 1e-3 | AdamW cos | - | tf 128/4 (+256/8, +3 blocks) | - | 0.37439 +- 0.00357 | - | 0.98 |
| v2 loss-norm / K9 / intent / idcg | 6 (6) | 2 | 1e-3 | AdamW cos | - | tf | - | 0.37982 +- 0.00132 | 0.29788 | 1.00 |
| v3-v7 cv inputs, GRU/DIN | 21 (16) | 2 | 1e-3 | AdamW cos | - | tf/gru/din | cv... | 0.38446 +- 0.00165 | 0.30885 | 1.00 |
| v8 cvnd 2/3/5 ep (+GRU/DIN/deep) | 10 (10) | 2-5 | 1e-3 | AdamW cos | - | tf/gru/din | cvnd | 0.38693 +- 0.00144 | 0.31245 | 0.97 |
| v9 cvnd 5/8/10 ep (+GRU/DIN/deep 5 ep) | 12 (12) | 5-10 | 1e-3 | AdamW cos | - | tf/gru/din | cvnd | 0.38960 +- 0.00191 | 0.31526 | 1.00 |
| v11 10-20 ep probes (gate/d256/deep/GRU) | 11 (0, killed 20-45k) | 10-20 | 1e-3 | AdamW cos | - | tf/gru, 128/256 | cvnd | - | - | - |
| stage-2 continuations (old recipe) | 54 (53) | 2-3 | 2e-4 | AdamW cos | - | tf | - | 0.38382 +- 0.00281 | 0.30630 | 1.00 |
| nnx optim / attn / feat 2ep arms | 48 (46) | 2 | 1e-3/2e-3 | AdamW or Muon, cos or wsd | 0/0.3 | tf | - | 0.3817-0.3838 | 0.3006-0.3029 | 0.97-1.00 |
| v16 / v18 pre2 / pre2s 3 ep pairs | 4 (4) | 3 | 1e-3 | AdamW cos | - | tf | pre2 / pre2s | 0.38841 / 0.38907 | 0.3131 | 0.95 |
| **F5** v12 (lr 2e-3, mix 0.3) | 6 (6) | 10 | 2e-3 | Muon WSD | 0.3 | tf 128/4 | cvnd/prew | 0.39247 +- 0.00094 | 0.31790 | 0.81 [0.61-0.81] |
| v13 hedges, seed 23 cvnd | 4 (4) | 10 | 1e-3/2e-3 | Muon WSD | 0/0.1/0.3 | tf | cvnd | 0.39344 +- 0.00131 | 0.31849 | 0.84 [0.46-0.86] |
| **F6** (lr 1e-3, mix 0.1) | 5+1 (6) | 10 | 1e-3 | Muon WSD | 0.1 | tf | cvnd | 0.39294 +- 0.00112 | 0.31795 | 0.76 [0.48-0.86] |
| v14_mix1_s91 | 1 (1) | 10 | 2e-3 | Muon WSD | 0.1 | tf | prew | 0.39428 | 0.31873 | 0.86 |
| **F5cd** 1-ep continuation | 6 (6) | 1 | 2e-3 | Muon, cd 1.0 | 0.1 | tf | cvnd/prew | 0.39427 +- 0.00087 | 0.31940 | 0.71 [0.71-1.00] |
| **F7** (lr 2e-3, mix 0.1) | 13 (7; 6 killed relaunches) | 10 | 2e-3 | Muon WSD | 0.1 | tf | pre2s | **0.39460 +- 0.00078** | **0.31911** | 0.86 [0.81-0.89] |
| v19 pre2s mix 0 | 6 (0, stopped at 5k) | 10 | 1e-3 | Muon WSD | 0 | tf | pre2s | - | - | - |
| **F8** 2 ep | 12 (12) | 2 | 1e-3 | Muon WSD | 0.1 | tf | pre2s | 0.39322 +- 0.00073 | 0.31775 | 1.00 |
| solvers 2-ep harness arms (sx_*) | 12 (10) | 2 | 1e-3 | Muon WSD | 0.1 | tf | pre2s | 0.39337 +- 0.00089 | 0.31719 | 1.00 [0.97-1.00] |
| **F6cd** 1-ep continuation | 3 (3) | 1 | 1e-3 | Muon, cd 1.0 | 0.1 | tf | cvnd | 0.39371 +- 0.00053 | 0.31797 | 0.51 [0.20-0.61] |
| **F9** (lr 1e-3, mix 0.1) | 6 (6) | 10 | 1e-3 | Muon WSD | 0.1 | tf | pre2s | 0.39382 +- 0.00115 | 0.31845 | 0.78 [0.51-0.86] |
| F10 GRU / F11 DIN / F12 d192 (running / queued) | 4 / 4 / 0 | 10 | 2e-3 | Muon WSD | 0.1 | gru / din / tf 192/6 | pre2s | partial curves, section 3.5 | | |

---------------------------------------------------------------------------------------------------------------------

## 2. Variation map: which knobs were ever varied

Counts over the 285 segments. "Muon+WSD" = the 102 segments of the current optimiser.

| knob | values over all runs (count) | values under Muon+WSD | status |
|---|---|---|---|
| epochs | 2 (176), 10 (70), 3 (14), 1 (9), 5 (7), 8 (2), 4, 15, 20 (killed) | 10 (56), 2 (33), 1 (9 continuations) | varied: 2 vs 10 (matched, F8 vs F9); 5 never run under Muon |
| lr | 1e-3 (178), 2e-3 (52), 2e-4 (55, stage 2) | 1e-3 (52), 2e-3 (50) | varied: 1e-3 vs 2e-3; never > 2e-3 |
| cooldown_frac | 0.2 (117), 1.0 (9 continuations) | 0.2 / 1.0 | **never varied in a full run** |
| cooldown_mix | 0.1 (78), 0 (29), 0.3 (19) | 0.1 (78), 0.3 (17), 0 (7; 6 stopped at 5k) | varied: 0 / 0.1 / 0.3 (one complete mix-0 10-ep run) |
| aux_anneal | 0.1 (104) | 0.1 only | **never varied** |
| aux_pos / aux_tau / aux_mode | 0.5 / 10 / listwise (+ 1.0, 30, PL) | 2-ep sx arms only | varied (2-ep): tau 30 worse, PL null, 1.0 null |
| retr_m | 0 (192), 16 (93) | 16 (83), 0 (19) | only 0 vs 16 |
| attn_gate | off / on | on | varied (nnx) |
| arch | tf (259), gru (16), din (10) | tf (92), gru (5), din (5) running | varied |
| d / heads | 128/4 (280); 256/8 (2, killed or old); 192/6 (smoke) | 128/4; d192/6 queued | width: old-recipe probes only |
| blocks / hist_layers | 2/1 (281); 3/2 (4, old recipe) | 2/1 only | **never under Muon** |
| ff | 2 (283) | 2 | **never varied** |
| dropout | 0.1 (285) | 0.1 | **never varied** (fine-tune: never varied either) |
| weight decay | 0.02 (285) | 0.02 | **never varied** (fine-tune 0.02 too) |
| warmup | 500 (221), 100 (stage 2), 0 (continuations) | 500 / 0 | never varied in from-scratch runs |
| cand_budget / max_feeds | 32768 / 512 (285) | same | **never varied** |
| max_hist | 256 (285) | 256 | = data cap; can only shrink |
| loss / loss_norm | softmax/feed (+ bce, both, batch, idcg in 2-ep old) | softmax/feed (3 batch) | varied (old); losses closed (sol_solvers) |
| xt preset | none / cv / cvix / cvhn / cvraw / cvnd / prew / pre2 / pre2s | cvnd / prew / pre2s | varied |
| xt_drop | 0 (282), 0.15, 0.3 | 0 | varied (old, closed) |
| Muon momentum / AdamW betas / clip / lr floor | hard-coded 0.95 nesterov / (0.9, 0.98) / 1.0 / 3% (nn/train.py:226,229,247,311) | same | **never varied** |
| fine-tune lr / epochs / EMA / batch / extra_lr_mult | lr 5e-5 (231), 1e-4 (15, old), 1e-3 (1); epochs 4 (125) / 6 (119); EMA 0.99 (116); 8192/128 (247); mult 10 (240) / 3 (7) | fta: lr 5e-5, 4 ep, EMA 0.99 | lr and batch never varied under fta; epochs / snapshot vs EMA varied (OVERFIT-4: mix-level ±0.0004) |

---------------------------------------------------------------------------------------------------------------------

## 3. Meta-analysis: matched comparisons only

Source: `nn_x/r6/recap/paired_report.txt` (from `paired.py`).
- Single-member comparisons use the holdout-selected checkpoints' stored TO predictions.
- "run-SE" = Welch SE from the seed spread, the honest one for family-vs-family. "feed-SE" = per-feed paired SE of the member-averaged per-feed NDCG, which ignores seed variance (a lower bound).

### 3.1 Learning rate (Muon, WSD 0.2, mix 0.1)
| comparison | val | holdout |
|---|---|---|
| F7 (2e-3, 7 seeds) vs F9 (1e-3, 6 seeds), both pre2s, 10 ep | **+0.00078** (run-SE 0.00055, z 1.4); M<50 +0.0010, M>=50 +0.0007, Tue +0.0005, Thu +0.0011 | **+0.00067** (run-SE 0.00023, z 2.9) |
| same seed 23, cvnd: v13_lr2_mix1 vs v13_lr1_mix1 | +0.00014 (feed-SE 0.0010) | +0.00069 |
| 6-seed TO ensembles F7 vs F9 | 0.39871 vs 0.39820 (+0.0005) | 0.32263 vs 0.32213 (+0.0005) |
| 2-ep nnx (cosine / WSD, AdamW-era base) | cosine +0.0008 (n.s.), **WSD -0.0010** (both seeds) | +0.0006-0.0007 |

**Reading.** The best lr depends on the horizon: lr 2e-3 loses at 2 epochs and wins at 10 epochs on both metrics (holdout z 2.9).
At 10 epochs lr 2e-3 also has the smaller late-cooldown decline (section 4.2). lr > 2e-3 was never run.

### 3.2 Training length (Muon, lr 1e-3, mix 0.1, pre2s)
| comparison | val | holdout |
|---|---|---|
| F9 (10 ep, 6 seeds) vs F8 (2 ep, 12 seeds), single member | +0.00060 (run-SE 0.00051); M<50 **+0.0019**, M>=50 -0.0003 | **+0.00069** (run-SE 0.00017, z 4.0) |
| 6-seed TO ensembles F9 vs F8 | **0.39820 vs 0.39418 (+0.0040)** | **0.32213 vs 0.31903 (+0.0031)** |
| AdamW cosine ladder (cvnd; v8/v9 families) | 2 ep 0.3863 -> 3 ep 0.3872 -> 5 ep 0.3893 -> 8 ep 0.3899 -> 10 ep 0.3913 | 0.3117 -> 0.3127 -> 0.3144 -> 0.3161 -> 0.3165 |

**Reading.**
- Under Muon + WSD, extra epochs buy little single-member strength (+0.0006 to +0.0007). They do buy most of the ensemble value: seed correlation is 0.978 at 2 epochs vs 0.938 at 10.
- The CURRENT_STATUS 21:20 comparison "2 ep ≈ 10 ep" (sx_base vs v13_lr1_mix1) was input-confounded (pre2s vs cvnd). The matched F9-vs-F8 pair shows a real single-member gain at 10 epochs, entirely on M<50 for val.

### 3.3 Cooldown regime
Seed 23, cvnd, identical up to the cooldown within an lr (`curves_report.txt`, last block). Columns are val / holdout.

| arm | 77.5k (pre-cd) | peak in cooldown | end (98.35k) | end - peak (val) |
|---|---|---|---|---|
| lr 1e-3 mix 0 | 0.3922 / 0.3175 | 85k 0.3943 / 0.3185 | 0.3936 / 0.3175 | -0.0007 |
| lr 1e-3 mix 0.1 | 0.3932 / 0.3179 | 82.5k 0.3939 / 0.3183 | 0.3896 / 0.3134 | -0.0043 |
| lr 1e-3 mix 0.3 | 0.3926 / 0.3177 | 77.5k (none) | 0.3651 / 0.2915 | collapse |
| lr 2e-3 mix 0.1 | 0.3927 / 0.3167 | 87.5k 0.3946 / 82.5k 0.3190 | 0.3928 / 0.3160 | -0.0018 |
| lr 2e-3 mix 0.3 (v12_all_e10) | 0.3927 / 0.3171 | 80k 0.3913 / 0.3172 | 0.3643 / 0.2893 | collapse |
| AdamW cosine (v9_e10, no retr/gate) | 0.3911 / 0.3160 | 85k 0.3921 / 0.3165 | 0.3914 / 0.3167 | -0.0007 |

Pooled over all complete 10-epoch Muon+WSD runs (end minus selected step):

| group | n | val | holdout | selected inside the cooldown |
|---|---|---|---|---|
| mix 0 | 1 | -0.0007 | -0.0009 | 1/1 |
| mix 0.1 | 21 | -0.0033 | -0.0041 | 14/21 |
| ↳ lr 1e-3 | 12 | -0.0043 | -0.0049 | 5/12 |
| ↳ lr 2e-3 | 9 | -0.0020 | -0.0031 | 9/9 |
| mix 0.3 | 7 | -0.027 | -0.028 | 4/7 |

2-epoch Muon+WSD runs (29, all mixes): 26/29 select the last step, 29/29 one of the last two evals; mean last minus selected is -0.00003.

Paired single-member comparisons:
- mix 0 vs mix 0.1 (seed 23, lr 1e-3): val +0.00045, holdout +0.00016. One seed, feed-SE 0.0010, so noise.
- mix 0.1 vs mix 0.3 (seed 23, lr 2e-3): val +0.0027, holdout +0.0017.
- 1-epoch continuations: F5cd vs F5 **+0.0018 val (run z 3.5) / +0.0015 holdout (z 5.0)**; F6cd vs F6 base +0.0013 / +0.0003.

**Reading.**
- The collapse and the late decline are pool memorisation, and the amount depends on the dose.
  - The holdout comes from the same random-like population as the pool but is disjoint, and it falls with val.
  - Mix 0 removes most of the decline. AdamW cosine without a pool shows the same mild -0.0007.
- The logged training loss cannot show the memorisation: the logged loss includes the annealed aux term (0.5 -> 0.1). Also, pool minus regular cooldown loss is -0.27 at 2 epochs (F8), where there is no decline, and -0.31 at 10 epochs (F9).
- Only the val/holdout curves are evidence.
- lr 2e-3 halves the decline. Two explanations are consistent with this:
  - the noise floor at the same schedule fraction is higher;
  - decoupled weight decay per step is lr·wd = 4e-5 vs 2e-5, i.e. a timescale of 1/(lr·wd) = 25k vs 50k steps.
  - Both point at regularisation strength in the cooldown (P2).

### 3.4 Inputs (train-stage xt)
- pre2s vs cvnd: 3-epoch pair +0.0019 val / +0.0002 holdout (CURRENT_STATUS 18:30).
- At 10 epochs, F9 vs F6: +0.0009 val (M<50 +0.0020, M>=50 +0.0001) / +0.0005 holdout (z_run 2.6).
- The 10-epoch input gain is about half the 3-epoch one, and still concentrated on M<50.
- prew 3-epoch: +0.0011 / +0.0008 (P2 in REVIEW_SYNTHESIS: net ~0 after the fine-tune).

### 3.5 Architecture and width (matched steps, same seed where available; `curves.jsonl`)
| comparison | evals | mean Δval | mean Δholdout | source |
|---|---|---|---|---|
| GRU vs tf, 3 ep (v8) | 30 | +0.0008 | +0.0000 | v8_nd_gru_e3 vs v8_nd_e3 |
| GRU vs tf, 5 ep (v9) | 25 | +0.0003 | +0.0004 | v9_gru_e5_s52 vs v9_e5_s51 |
| GRU vs tf, 10 ep cosine (to 32-35k) | 13-14 | +0.0007 / +0.0008 | +0.0004 / +0.0001 | v11_gru_e10_s71/72 vs v9_e10 |
| **F10 GRU s501 vs F7 mean (to 37.5k, live)** | 15 | +0.0004 | +0.0004 | v23_gru_pre2s_s501 |
| deep (3 blocks + 2 hist layers) vs tf, 3 / 5 / 10 ep | 30 / 25 / 14 | -0.0000 / +0.0002 / +0.0001 | -0.0002 / +0.0003 / -0.0003 | v8 / v9 / v11 deep |
| DIN vs tf, 3 / 5 ep | 30 / 25 | **-0.0012 / -0.0026** | -0.0018 / -0.0021 | v8 / v9 din |
| **F11 DIN s601 vs F7 mean (to 67.5k, live)** | 27 | **-0.0018** | -0.0013 | v24_din_pre2s_s601 |
| d256/8h vs d128 at 10 ep cosine (to 27.5k) | 11 | -0.0000 | -0.0004 | v11_d256_e10 vs v9_e10 |
| heads 8 / blocks 3 / d256 at 2 ep (round 1, best val only) | - | -0.0011 / -0.0006 / -0.0008 | - | aux_sim_h8 / aux_sim_b3 / big_nopos |
| retr16+gate vs base, 10 ep cosine (to 25k) | 10 | **+0.0028** | **+0.0030** | v12_rg_e10 vs v9_e10 |
| Muon+WSD vs AdamW, 10 ep (to 30k, plateau) | 12 | +0.0012 | +0.0010 | v12_muwsd_e10 vs v9_e10 |

**Reading.**
- Capacity (d, depth, heads) is flat in every recipe where it was probed, but never under Muon (F12 will be the first).
- GRU is a same-strength alternative history encoder.
- DIN is reliably 0.001-0.003 weaker. On current evidence F11 is borderline for the 0.002 strength gate.

### 3.6 Ensemble value by recipe (TO, z-mean of the first k seeds; `paired_report.txt`)
| family | single (mean) | k=2 | k=3 | k=6 | gain 1->6 | holdout single -> k=6 | within-feed seed corr (mean, sd, n pairs) |
|---|---|---|---|---|---|---|---|
| AdamW cos 10 ep (v9_e10 x4) | 0.39127 | | | 0.39281 (k=4) | +0.0015 (k=4) | 0.31652 -> 0.31810 (k=4) | 0.967 |
| F5 | 0.39247 | 0.39411 | 0.39522 | 0.39712 | +0.0047 | 0.31790 -> 0.32162 | 0.938 (0.003, 15) |
| F6 | 0.39294 | 0.39565 | 0.39611 | 0.39697 | +0.0040 | 0.31795 -> 0.32183 | 0.930 (0.005, 15) |
| F7 | 0.39471 | 0.39695 | 0.39814 | **0.39871** | +0.0040 | 0.31913 -> **0.32263** | 0.941 (0.004, 21) |
| F9 | 0.39382 | 0.39622 | 0.39691 | 0.39820 | +0.0044 | 0.31844 -> 0.32213 | 0.938 (0.007, 15) |
| F8 (2 ep) | 0.39318 | 0.39411 | 0.39448 | 0.39418 | +0.0010 | 0.31767 -> 0.31903 | **0.978 (0.0005, 66)** |
| F5cd (cooled twins of F5) | 0.39426 | 0.39505 | 0.39632 | 0.39702 | +0.0028 | 0.31940 -> 0.32279 | 0.947 (0.002, 15) |

Cross-family correlation of the 6-seed TO ensembles is 0.978-0.986.
12-member pools: F7+F9 val 0.39906 / holdout 0.32288; F6+F9 0.39873 / 0.32243; F7+F6 0.39873 / 0.32278.

**Reading.**
- Cooling raises seed correlation (F5cd 0.947 vs F5 0.938).
- Short training raises it a lot (F8 0.978). The AdamW cosine anneal raises it too (0.967).
- The diverse members are the ones that spent a long time at a high constant lr and are taken while still noisy. Averaging a run's plateau iterates removes this: avg3 members correlate 0.975 across seeds (section 5).
- F9 passes the single-member gate (0.39382 / 0.31844 vs F7 0.39460 / 0.31911) and the 6-seed gate (b) (0.39820 / 0.32213 vs 0.39871 / 0.32263), both within 0.002. This is information for the lead's pre-declared decision.

### 3.7 Fine-tune curves (52 n2c_pre_v3 cos+EMA fine-tunes; `ft_runs.csv`)
- Mean OOF gain over the TO epoch-0 by epoch: +0.00126 (ep 1), +0.00100, +0.00090, +0.00085 (ep 4).
- The best epoch is 1 in 37/52 runs. The final EMA used in the files gives +0.00091.
- By family, EMA gain: F7 +0.00142, F5 +0.00158, F6 +0.00134, F5cd +0.00145, **F9 +0.00078**, F6cd +0.00007, **F8 -0.00015**.
- The fine-tune over-trains after epoch 1, and adds least to the models trained at lr 1e-3 and for 2 epochs.
- The mix-level effect of a shorter fine-tune was already refuted (REVIEW_SYNTHESIS 2, OVERFIT-4: ±0.0004 in the 0.5/0.5 mix). Not re-proposed above "low".

---------------------------------------------------------------------------------------------------------------------

## 4. Training-curve shapes

Family mean curves (val / holdout) are in `nn_x/r6/recap/curves_report.txt`. Key points:

| family | 20k | 40k | 60k | 77.5k (pre-cd) | 82.5k | 87.5k | 92.5k | end |
|---|---|---|---|---|---|---|---|---|
| F7 (lr 2e-3) val | 0.38990 | 0.39185 | 0.39284 | 0.39212 | 0.39432 | 0.39475 | 0.39300 | 0.39260 |
| F7 holdout | 0.31489 | 0.31596 | 0.31649 | 0.31732 | 0.31844 | 0.31837 | 0.31671 | 0.31602 |
| F9 (lr 1e-3) val | 0.39122 | 0.39224 | 0.39364 | 0.39381 | 0.39358 | 0.39306 | 0.39043 | 0.38944 |
| F9 holdout | 0.31568 | 0.31705 | 0.31738 | 0.31808 | 0.31758 | 0.31661 | 0.31433 | 0.31312 |
| F8 (2 ep) val (steps 4k / 8k / 12k / 15k / 17k / 18k / 19k / end) | 0.38513 | 0.38767 | 0.38950 | 0.38993 | 0.39173 | 0.39258 | 0.39294 | 0.39322 |
| F8 holdout | 0.30986 | 0.31261 | 0.31384 | 0.31470 | 0.31620 | 0.31700 | 0.31742 | 0.31775 |

1. **Where the gain arrives.**
   - 10 epochs: by 20k steps (2 epochs at constant lr) val is already within 0.0024-0.0047 of the selected checkpoint.
     - The rest of the plateau (20k -> 77.5k) adds +0.0022-0.0026 val / +0.0020-0.0024 holdout.
     - The cooldown adds at most +0.0025 val (F7, pre-cd -> peak) or +0.0005 (F9).
   - 2 epochs: the cooldown adds +0.0033 val / +0.0031 holdout (15k -> end) and is still rising at the last step.
2. **Plateau noise.**
   - Within-run val jitter around a linear trend over 40-77.5k: sd 0.00086 (F6), 0.00081 (F9), 0.00074 (F7).
   - The mean |pairwise| difference between seeds at the same step is 0.0010 / 0.0005 (val / holdout).
   - The plateau iterate is a noisy draw around a slowly rising mean.
   - The gradient-noise scale (McCandlish et al. 2018) at plateau checkpoints, on the per-feed mean-normalised training loss (main CE + 0.5 (1-p) K9), dropout on.
     - Final numbers (`nn_x/r6/logs/recap_gns2.log`; 192 batches x 128 = 24,576 train feeds per checkpoint; jackknife SE over 8 groups):

       | checkpoint | B_simple, all params (feeds) | split halves | Muon matrices only |
       |---|---|---|---|
       | F9 s403 (77.5k, lr 1e-3) | **4,936 +- 801** | 5,010 / 4,513 | 8,742 +- 1,218 |
       | F7 s101@77.5k (lr 2e-3) | **9,393 +- 3,099** | 5,328 / 19,276 | 19,835 +- 1,672 |

     - The first, smaller pass (3,072 feeds, `recap_gns.log`) gave the same order (7K-19K) but was too noisy: |G|^2 was 70-90% noise. On cooled checkpoints (F8 s301) |G|^2 is below the noise floor, as expected near a cooled minimum.
   - The training batch is ~317 feeds, 15-30x below the noise scale, so the plateau is in the noise-dominated regime.
     - Pre-declared reading (B_simple >= 2 x 317): a 2x batch costs only E/E_min = 1 + B/B_noise ≈ +4-7% more examples for the same loss, i.e. nearly proportional per-step progress.
     - Section 5 shows that this same noise is the ensemble's diversity, so removing it (a larger batch at the same lr, lr decay, averaging) is not free at ensemble level.
3. **Selected-step distribution.**
   - lr 2e-3, mix 0.1: 9/9 complete runs selected in the first 1-9k cooldown steps (80-87.5k).
   - lr 2e-3, mix 0.3 (F5): 4/6 at 80k (the first cooldown eval), 2/6 on the plateau (60k, 70k).
   - lr 1e-3, mix 0.1: 5/12 in the cooldown (all <= 85k), 7/12 on the plateau (47.5-77.5k).
   - Continuations: 60-80% of the way through (F5cd), 20-61% (F6cd).
   - 2 epochs: the last step.
   - The holdout checkpoint selection is a max over ~40 noisy evals. The holdout value at the selected step is therefore optimistic. Val at that step is not: val labels play no part in the selection, so it is an unbiased score of that checkpoint. Because iterate quality fluctuates by ~one jitter sd, the selection does pick genuinely better-than-average plateau iterates.
4. **Over-fitting signature.** Only in runs with >= 8 data passes before the cooldown, and only once the lr falls below ~0.7-1e-3 with pool replay on (section 3.3). No over-fitting appears on the plateau itself: val and holdout keep creeping up to 77.5k in every family.
5. **Run-to-run noise.**

   | family | single-member val sd | holdout sd |
   |---|---|---|
   | F5, F6, F7, F9 | 0.0008-0.0011 | 0.0003-0.0005 |
   | F8 | 0.0007 | 0.0003 |
   | AdamW cosine 10 ep | 0.0004 | 0.0002 |

   - Same-seed, same-config non-determinism: v13 lr 1e-3 trio and v12_all_e10/v13_lr2_mix1 pair.

     | phase | mean pairwise \|Δval\| | mean pairwise \|Δhold\| |
     |---|---|---|
     | <= 10k steps | 0.0002-0.0005 | 0.0001-0.0002 |
     | 10-40k | 0.0003-0.0005 | 0.0003 |
     | 40-77.5k | 0.0007 | 0.0003 |

   - Different seeds: 0.0010 / 0.0005.
   - MDE for a 6-vs-6 family comparison at 80% power is ≈ 2.8·sqrt(2/6)·0.0010 ≈ 0.0016 val and 0.0006 holdout.

---------------------------------------------------------------------------------------------------------------------

## 5. New CPU evidence: weight averaging, member noise and the ensemble-level (mean) function

Motivation:
- The plateau is noise-dominated (section 4.2), and the late cooldown memorises (section 3.3).
- Averaging iterates removes the noise without lowering the lr. This is the classic LAWA / SWA / checkpoint-merging argument (Izmailov et al. 2018; Kaddour 2022; Sanyal et al. 2023; WSM, arXiv 2507.17634, cited in nnx_research_llm_arch.md:158).
- C1 (stage-1 EMA) was closed on the 2-epoch **cosine** baseline (nnx_VERDICT 5), where the anneal already averages. The 10-epoch WSD plateau was never tested.

Method:
- `nn_x/r6/recap_swa.py` (probe, seeds 101/102) and `nn_x/r6/recap_avg7.py` (all 7 F7 seeds), CPU fp32 inference.
- avg3 = uniform mean of the state_dicts of `v17_pre2s_s{s}@60000`, `@70000`, `@77500` (all plateau, lr 2e-3), compared with the same run's holdout-selected cooled checkpoint `v17_pre2s_s{s}`. Both are scored with the same CPU pipeline.
- Rules pre-declared in the script headers before running.

Probe (recap_swa.log):

| s101 | val |
|---|---|
| @60k | 0.39272 |
| @70k | 0.39308 |
| @77.5k | 0.39149 |
| cooled, holdout-selected (80k) | 0.39365 |
| **avg3** | **0.39601** |
| avg2 (70k, 77.5k) | 0.39448 |

- avg3 - cooled **+0.00235 +- 0.00086**; avg3 - best snapshot +0.00293 +- 0.00076; avg3 - @77.5k +0.00452 +- 0.00072.
- Both pre-declared thresholds pass for s101: > +0.0010 over the best snapshot, and >= cooled - 0.0005.
- s102: @60k 0.39342, @70k 0.39297, @77.5k 0.39254, cooled (85k) 0.39526, **avg3 0.39532**, avg2 0.39500.
  - avg3 - cooled +0.00006 +- 0.00085; avg3 - best snapshot +0.00190 +- 0.00074.
  - Both thresholds pass, but s102's cooled checkpoint was already the second-best F7 member.
- Reading of the probe: averaging always beats every single plateau snapshot (+0.0019 to +0.0029). Against the cooled checkpoint it ranges from a tie to +0.0024, so the 7-seed run decides.

**7-seed result** (`nn_x/r6/recap/avg7_eval.txt`; all CPU fp32; reading pre-declared in recap_avg7.py):

| seed | avg3 val / holdout | cooled val / holdout | Δ val | Δ holdout |
|---|---|---|---|---|
| s101 | 0.39601 / 0.32011 | 0.39365 / 0.31847 | +0.00235 | +0.00164 |
| s102 | 0.39532 / 0.32019 | 0.39526 / 0.31957 | +0.00006 | +0.00062 |
| s103 | 0.39452 / 0.32006 | 0.39446 / 0.31844 | +0.00006 | +0.00162 |
| s104 | 0.39446 / 0.31935 | 0.39578 / 0.31979 | -0.00132 | -0.00044 |
| s105 | 0.39492 / 0.31897 | 0.39418 / 0.31856 | +0.00074 | +0.00041 |
| s106 | 0.39522 / 0.32003 | 0.39494 / 0.31928 | +0.00028 | +0.00075 |
| s107 | 0.39454 / 0.31942 | 0.39382 / 0.31898 | +0.00071 | +0.00044 |
| **pooled** | | | **+0.00041 +- 0.00040** (z 1.0; M<50 -0.0005, M>=50 +0.0010, Tue +0.0004, Thu +0.0004) | **+0.00072** (feed-SE 0.00024, calibrated 0.00040) |

The CPU fp32 cooled numbers match the stored GPU-bf16 ones within 0.0002 (e.g. s104 0.39578 vs 0.39568).

| 7-member TO ensemble | val | holdout | within-feed seed corr |
|---|---|---|---|
| cooled (= F7 TO) | **0.39866** | **0.32279** | 0.9406 (sd 0.0043) |
| avg3 | 0.39618 (**-0.00247 +- 0.00055**) | 0.32115 (**-0.00164 +- 0.00032**) | **0.9752** (sd 0.0014) |
| 14-member pool (avg3 + cooled) | 0.39768 (-0.00098 +- 0.00037) | 0.32231 (-0.00048) | - |

avg3 vs its own cooled checkpoint correlate 0.966.

**Verdict (pre-declared rule):**
- Single member: NOT confirmed (val z 1.0 < 2; holdout positive; 6/7 seeds positive).
- Ensemble: averaged members are worse at every member count above one. P1 (weight averaging) is CLOSED for the pool; keep it only for a single-model use case.

**Mechanism (the important part).**
- Averaging a run's plateau iterates gives each seed nearly the same function. The avg3 cross-seed correlation is 0.975, the same as 2-epoch members (0.978) and close to the 0.978-0.986 correlation between whole-family ensembles (section 3.6).
- So most of the "seed diversity" of a 10-epoch member is its position in the noisy plateau random walk, not a different basin.
- An ensemble of cooled seeds averages that noise *in function space* and comes out 0.0025 better than averaging it in weight space.
- Consequences for every proposal below:
  - (i) Anything that removes member noise (weight averaging, a longer or lower cooldown, a larger batch at the same lr, short training) makes single members look better and the ensemble worse or unchanged.
  - (ii) The quantity that matters is the quality of the ensemble-level ("mean") function. Single-member screens at 10 epochs cannot see it: single sd 0.0008-0.0011 is mostly this noise.
  - (iii) 2-epoch members are low-noise (seed corr 0.978, k=6 gain only +0.0010), so a 2-epoch single member is close to the 2-epoch mean function. 2-epoch screens are therefore a good, cheap probe of mean-function changes, but only of the 2-epoch mean function: F9 vs F8 ensembles differ by +0.0040.
- The C1 closure (stage-1 EMA null on the cosine baseline) is consistent with this.

**Mean function along the plateau** (`nn_x/r6/recap_plateau_ens.py`, log `nn_x/r6/logs/recap_plateau_ens.log`; CPU fp32 val inference of all 21 F7 snapshots; reading pre-declared in the script header):

| 7-seed F7 members taken at | mean single val | 7-seed TO ensemble val | ensemble gain | seed corr |
|---|---|---|---|---|
| 60k (plateau, lr 2e-3) | 0.39271 | **0.39660** | +0.0039 | 0.940 |
| 70k (plateau) | 0.39220 | 0.39578 | +0.0036 | 0.934 |
| 77.5k (plateau end) | 0.39200 | 0.39551 | +0.0035 | 0.943 |
| cooled, holdout-selected (80-87.5k) | 0.39459 | **0.39866** | +0.0041 | 0.941 |
| all 21 plateau snapshots | | 0.39568 | | |
| 21 plateau + 7 cooled | | 0.39694 | | |

Paired differences of the 7-seed ensembles:

| comparison | Δ val |
|---|---|
| ens(77.5k) - ens(60k) | **-0.00108 +- 0.00058** |
| ens(77.5k) - ens(70k) | -0.00027 +- 0.00059 |
| **ens(cooled) - ens(77.5k)** | **+0.00314 +- 0.00063** |
| ens(all 21 snapshots) - ens(cooled) | -0.00298 +- 0.00057 |

**Reading (pre-declared thresholds).**
1. At lr 2e-3 the ensemble-level function stops improving by 60k steps (6 epochs). It is flat to slightly worse at 77.5k: -0.0011 is below the +0.0005 "saturated" threshold. **A longer plateau is not a lever.** The last ~1.8 epochs of the plateau buy nothing for the ensemble, and the seed diversity is already fully developed at 60k (corr 0.940).
2. **The early cooldown improves the mean function (+0.0031, z 5)**, not just member noise: seed correlation is unchanged (0.941 vs 0.943). Keep the cooldown. The gain comes while the lr falls from 2e-3 to ~1e-3. Below ~5e-4 pool memorisation degrades it (section 3.3).
3. The F9 6-seed ensemble (lr 1e-3 constant, mostly plateau-selected) reaches 0.39820, close to the F7 cooled ensemble's 0.39871. Together with point 2, this says **the ensemble-optimal operating point is an lr near 1e-3, with enough noise left for diversity**. A higher lr (2e-3) blurs the mean function, and a lower lr (< 5e-4, late cooldown) memorises the pool.
4. Corollary (proposal P1): since the plateau mean function saturates by 6 epochs, a run with the cooldown moved earlier should give the same ensemble at ~65-80% of the GPU cost. This can be tested cheaply with continuations from the existing @60000 snapshots.

Files (information only; given the ensemble verdict there is no reason to fine-tune them):
- averaged checkpoints `nn_x/r6/ckpt/r6avg3_s101..s107.pt` (nn/ckpt format);
- CPU fp32 predictions `nn_x/r6/preds/nn_{r6avg3,r6cool}_s*_{trainonly_val,holdout}.npy` and `nn_x/r6/preds/nn_r6snap_s*@{60000,70000,77500}_trainonly_val.npy`.

---------------------------------------------------------------------------------------------------------------------

## 6. Never-varied knobs ranked by plausible impact under the current recipe

Ranking principle, from section 5: the pool's quality is set by the ensemble-level ("mean") function. Member noise is the pool's diversity. A knob ranks high if it plausibly changes the mean function or the GPU cost per family. Knobs that only de-noise single members rank low.

| rank | knob(s) | evidence | expected effect | screen that can see it |
|---|---|---|---|---|
| 1 | **schedule length / cooldown start** (epochs 10 with cooldown at 80%; never shortened under Muon) | plateau mean function saturates by 60k (7-seed ens 60k 0.39660 vs 77.5k 0.39551); seed corr already 0.940 at 60k; the early cooldown adds +0.0031 to the mean function | same ensemble at ~65-80% GPU per member (more families per GPU-hour) | continuation from the existing @60000 snapshots, 7 seeds, ensemble level (P1) |
| 2 | **dropout 0.1, wd 0.02, aux_anneal 0.1, P8 aux norm, pool replay** (cooldown-phase regularisation) | late-cooldown memorisation in 19/21 mix-0.1 runs degrades the mean function; halved at 2x lr·wd; never varied in 285 runs | lets the cooldown push the mean function past the current peak: +0.0005..+0.0015 | continuations from shared plateau snapshots (P2); a 2-ep from-scratch screen is blind |
| 3 | **cand_budget / max_feeds (batch) with lr** | B_simple 4.9K-9.4K vs 317 feeds | ~0 quality at constant lr/B; 25-40% less GPU time if the GPU is not saturated | 2-ep screen for speed + single strength; 10-ep 4-seed ensemble confirm (P3) |
| 4 | **retr_m** (16 only) | A4 = strongest architecture item (+0.0019 val / +0.0033 hold); m never varied | ±0.0005 to the mean function | 2-ep screen (low-noise members ≈ 2-ep mean function) (P4) |
| 5 | blocks / hist_layers / heads / ff under Muon | capacity flat in the AdamW era; F12 d192 queued | ~0 | wait for F12; then 2-ep screen |
| 6 | lr > 2e-3 | lr 2e-3 plateau already blurs the mean function (section 5); the gain at 10 ep comes from the cooldown through ~1e-3 | likely ≤ 0 | skip |
| 7 | iterate averaging (weight space) | measured here: single +0.0004 (n.s.), ensemble -0.0025 | negative for the pool | closed |
| 8 | warmup, Muon momentum, AdamW betas, clip 1.0 | no instability anywhere | ~0 | skip |
| 9 | max_hist | 256 is the data cap | n/a | skip |
| 10 | fine-tune lr / epochs / EMA / extra_lr_mult / batch | epoch-1 peak in 37/52 runs; mix-level effect refuted (OVERFIT-4) | ≤ ±0.0004 in the mix | skip |

---------------------------------------------------------------------------------------------------------------------

## 7. Proposals (ranked), exact specs

All GPU work belongs to the NN agent / lead on the allowed GPUs (4-7) under the project rules. This recap ran no GPU job.
Base flags, identical to run_final7b.sh / run_final12.sh and called BASE below:
```
--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout
--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --sched wsd --cooldown_frac 0.2 --aux_anneal 0.1 --retr_m 16 --attn_gate
```
Ensemble-level evaluation everywhere: report the k-seed TO ensemble (val and holdout) and the within-feed seed correlation next to the single-member means. Never accept a recipe change on single-member gains alone (section 5).

### P1. Earlier cooldown: is a ~7-epoch schedule as good as 10 at ensemble level? (cost lever; cheap test on existing snapshots)
- **Evidence:** section 5, mean-function table. The 7-seed plateau ensemble at 60k ≥ at 77.5k; seed diversity is complete at 60k; the cooldown supplies the mean-function gain (+0.0031).
- **Test (no from-scratch training).**
  - 7 continuations `train.py --variant r6cd60_s10x --init v17_pre2s_s10x@60000 BASE --epochs 1 --cooldown_frac 1.0 --warmup 0 --lr 2e-3 --cooldown_mix 0.1 --eval_every 1000 --seed 10x`, x = 1..7.
  - That is 1 epoch of cooldown from the 6-epoch plateau, i.e. a 7-epoch run, like N2 / F5cd which peaked at ~70% of the continuation.
  - Cost ~7 x 15-20 GPU-min, then predict / holdout_eval / n2cpv3 fta as usual.
  - Optional second arm: the same from `@77500` (the F5cd analogue for F7).
- **Pre-declared acceptance:**
  - 7-member TO ensemble of r6cd60 >= F7 TO ensemble - 0.0005 on val (0.39866 CPU / 0.39871 GPU) and >= F7 - 0.0003 on holdout (0.32279 / 0.32263);
  - seed correlation <= 0.95.
  - If met, new families use `--epochs 7` (cooldown_frac 0.2 -> plateau 5.6 ep + cooldown 1.4 ep) or the continuation form. That is ~30% less GPU per member, so ~1.4x more families per GPU-hour, the lever SOLUTIONS_REVIEW names.
  - Information bonus: the r6cd60 members are same-input, different-schedule twins of F7. Report the 14-member pool (F7 + r6cd60) on the holdout, as a pre-declared extra candidate for the lead.
- **Expected:** ensemble within ±0.0005 of F7. Risk: the cooldown from a shorter plateau may need a different length.

### P2. Cooldown-phase regularisation screen (continuations from shared plateau snapshots)
- **Design.**
  - Start from `v17_pre2s_s101@77500` and `v17_pre2s_s102@77500`.
  - Each arm is a 2-epoch continuation reproducing the original cooldown length (19,670 steps): `train.py --init v17_pre2s_s10x@77500 BASE --epochs 2 --cooldown_frac 1.0 --warmup 0 --lr 2e-3 --cooldown_mix 0.1 --eval_every 1000 --seed 10x`.
  - Same start, same data order per seed, so pairing is strong: same-start divergence is ~0.0002-0.0005 in the first 10k steps (section 4.5).
  - The init path loads the architecture from the checkpoint cfg (nn/train.py:171-197), so `--drop` is ignored. The nn_x/r6 copy of train.py must override the Dropout modules' p and Attn.drop after loading.
  - It must also add `--aux_pool_norm` (P8 fix: aux denominator = number of feeds, so pool batches get (1-p) ≈ 0.05 weight; nn/train.py:307) and `--hist_drop p`, which masks a random fraction p of valid history events per feed in training: `hmask &= keep`, `hemb_ok &= keep` before `model.dense`.
  - Redirect ckpt / preds / logs to nn_x/r6.
- **Arms** (x2 starts):

  | arm | change |
  |---|---|
  | C0 | control; plus one same-start replicate C0r (s101) |
  | A1 | `--wd 0.1` |
  | A2 | dropout 0.2 |
  | A3 | `--aux_anneal 0.5` (no anneal) |
  | A4 | P8 fix |
  | A5 | `--cooldown_mix 0` |
  | A6 | `--hist_drop 0.15` |
  | A7 (optional) | larger cooldown pool, p_random > 0.2 (+47% distinct pool feeds; sol_solvers P3, needs a `--pool_p` flag in the copy) |

  - ~15 runs of 2 epochs each, ~25-30 GPU-min each alone.
  - Priority order: C0, A1, A2, A3, A5, then the rest.
- **Pre-declared acceptance per arm vs C0 (pooled over the 2 starts):**
  - (i) mechanism: the late decline (selected minus last, on val and holdout) shrinks by >= 50% in both starts;
  - (ii) outcome: holdout at the selected step >= C0 + 0.0005, val >= C0, both starts positive on holdout;
  - (iii) diversity: within-feed correlation with C0 from the same start is no higher than C0r's correlation with C0. That correlation is the noise-only reference, so this checks that the arm does not just de-noise.
  - Winners go to a 10-epoch (or P1-length) family of 4-6 seeds, judged by gates (a)+(b) and the holdout selector. A regularised family is also a same-strength different-recipe family.
- **Expected:** +0.0005..+0.0015 at the selected step if memorisation is what caps the cooldown.

### P3. Batch x2 at constant noise temperature (throughput lever)
- **Evidence:** B_simple 4,900-9,400 >> 317 feeds (section 4.2); cand_budget / max_feeds never varied (section 2).
- **Why constant temperature:** member noise is the ensemble's diversity (section 5). Keeping lr/B constant preserves the plateau's stationary noise level.
- **Screen (2-ep harness = F8 recipe, seeds 23 / 24 like sx_base):**
  - `BASE --epochs 2 --lr {2e-3 | 1.4e-3} --cooldown_mix 0.1 --cand_budget 65536 --max_feeds 1024 --eval_every 500 --seed {23,24}`.
  - Reference: F8 + sx_base, 14 runs: val 0.39323 sd 0.00074, holdout ~0.31773 sd 0.00028.
  - Also log minutes per epoch.
- **Pre-declared acceptance:**
  - arm mean holdout >= ref - 0.0004 and val >= ref - 0.0011 (2 SE non-inferiority with 2 vs 14 runs);
  - wall-clock <= 0.75x;
  - the 2-seed within-feed correlation is reported against F8's 0.978 +- 0.0005.
- **If it passes:** a 10-epoch (or P1-length) 4-seed confirm at lr x2 with k=4 TO ensemble and seed correlation vs F7 / F9.
  - Adopt only if the k=4 ensemble is within 0.001 of the matched family on val and holdout and the seed correlation <= 0.95.
  - Gain: more families per GPU-hour, not a direct NDCG gain. It stacks with P1.

### P4. retr_m 32 (and 8 as a bracket)
- 2-ep harness: `BASE --epochs 2 --lr 1e-3 --cooldown_mix 0.1 --retr_m {8,32} --seed {23,24}`, paired with sx_base_s23 / s24.
- 2-ep members are low-noise (seed corr 0.978), so a single member ≈ the 2-ep mean function and the screen measures a mean-function change directly.
- Acceptance: paired val > 2 SE over 2 seeds and holdout >= 0 (unified rule). Expected ±0.0005. Cost: 4 runs x ~25 min.
- A winner needs a 10-epoch ensemble-level confirm, because the 2-ep mean function differs from the 10-ep one (+0.0040).

### Closed by this recap / not recommended now
- **Plateau weight averaging** (avg3) as members: ensemble -0.0025 val / -0.0016 holdout (section 5). Closed for the pool.
- **Longer plateau** (12-15 epochs): the plateau mean function is saturated by 6 epochs at lr 2e-3 (section 5).
- **lr > 2e-3:** the lr 2e-3 plateau already blurs the mean function; low prior.
- **5-epoch family:** superseded by P1, which tests the same cost lever on existing snapshots.
- Capacity knobs (ff 4, blocks 3, hist_layers 2, heads 8): flat in every probe; wait for F12 (d192).
- warmup / betas / momentum / clip: no instability evidence.
- Fine-tune knobs: mix-level refuted.
- More 2-epoch same-recipe seeds: closed in SOLUTIONS_REVIEW.

---------------------------------------------------------------------------------------------------------------------

## 8. Caveats
- The holdout was used for every member's checkpoint selection (~40 evals per run). Holdout differences between selected checkpoints are optimistic. Section 4.5 gives the run-level noise.
- The avg3-vs-cooled comparison is not affected: avg3 involves no selection, and the cooled checkpoint is the selected one, so any selection bias works against avg3.
- The mean-function analysis (section 5) uses one family (F7, lr 2e-3) and val only. Plateau snapshot holdout predictions exist only for @77.5k. The saturation-by-60k statement is therefore F7-specific; F9 (lr 1e-3) has no snapshots.
- The final GNS estimates use 24,576 feeds per checkpoint (noise share of |Gbar|^2 is 0.17-0.28; jackknife SE 16-33%). The first 3,072-feed pass was too noisy to quote. B_simple is computed on all parameters. Muon's orthogonalised update does not map exactly onto the SGD critical-batch theory, so read it as an order-of-magnitude statement (15-30x the batch).
- The GNS uses mixed-M random batches. Training batches are M-bucketed (nn/data.py:110-127), with 512/B aux weighting (P8).
- CPU fp32 vs GPU bf16 inference moves single-model val by up to ~0.0003 (s101 cooled: 0.39365 CPU vs 0.39361 GPU). All avg3-vs-cooled comparisons use the same CPU pipeline on both sides.
- Old-recipe architecture comparisons are single-seed and step-matched curves (no stored predictions for killed runs). Read them as ±0.0005.
- F10 / F11 / F12 were still running at writing time. Their rows use partial curves only.

---------------------------------------------------------------------------------------------------------------------

## Appendix A. Every complete Muon+WSD run (67 rows; from nn_x/r6/recap/runs.csv)
The selected step is holdout-selected. Continuations (v15_cd_*, v21_cd_*) take architecture and inputs from their --init checkpoint (retr16 + gate + cvnd/prew). The full 285-row table with all hyperparameters is in runs.csv.

| variant | ep | lr | mix | retr/gate | arch d/h | xt | seed | sel step | TO val @sel | hold @sel | max val (step) | last val / hold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v15_cd_all_e10 | 1 | 0.002 | 0.1 | from init | tf 128/4 | init v12_all_e10 | 23 | 7000 | 0.39318 | 0.31926 | 0.39353 (6000) | 0.39257 / 0.31863 |
| v15_cd_all_s81 | 1 | 0.002 | 0.1 | from init | tf 128/4 | init v12_all_s81 | 23 | 7000 | 0.39343 | 0.31858 | 0.39343 (7000) | 0.39313 / 0.31831 |
| v15_cd_all_s82 | 1 | 0.002 | 0.1 | from init | tf 128/4 | init v12_all_s82 | 23 | 8000 | 0.39466 | 0.31951 | 0.39524 (6000) | 0.39476 / 0.31929 |
| v15_cd_allprew_s83 | 1 | 0.002 | 0.1 | from init | tf 128/4 | init v12_allprew_s83 | 23 | 9835 | 0.39426 | 0.32013 | 0.39471 (8000) | 0.39426 / 0.32013 |
| v15_cd_allprew_s84 | 1 | 0.002 | 0.1 | from init | tf 128/4 | init v12_allprew_s84 | 23 | 7000 | 0.39451 | 0.31905 | 0.39472 (5000) | 0.39439 / 0.31853 |
| v15_cd_allprew_s85 | 1 | 0.002 | 0.1 | from init | tf 128/4 | init v12_allprew_s85 | 23 | 7000 | 0.39555 | 0.31989 | 0.39555 (7000) | 0.39480 / 0.31928 |
| v21_cd_v14_s92 | 1 | 0.001 | 0.1 | from init | tf 128/4 | init v14_s92 | 92 | 5000 | 0.39353 | 0.31869 | 0.39383 (6000) | 0.39191 / 0.31746 |
| v21_cd_v14_s94 | 1 | 0.001 | 0.1 | from init | tf 128/4 | init v14_s94 | 94 | 6000 | 0.39431 | 0.31753 | 0.39431 (6000) | 0.39303 / 0.31634 |
| v21_cd_v14_s96 | 1 | 0.001 | 0.1 | from init | tf 128/4 | init v14_s96 | 96 | 2000 | 0.39329 | 0.31769 | 0.39329 (2000) | 0.39223 / 0.31716 |
| optim_mu2wsd_s23 | 2 | 0.002 | 0.3 | 0/F | tf 128/4 | - | 23 | 19670 | 0.38500 | 0.30462 | 0.38500 (19670) | 0.38500 / 0.30462 |
| optim_mu2wsd_s24 | 2 | 0.002 | 0.3 | 0/F | tf 128/4 | - | 24 | 19670 | 0.38483 | 0.30442 | 0.38483 (19670) | 0.38483 / 0.30442 |
| optim_mu2wsdkd_s23 | 2 | 0.002 | 0.3 | 0/F | tf 128/4 | - | 23 | 19670 | 0.38525 | 0.30437 | 0.38597 (19000) | 0.38525 / 0.30437 |
| optim_mu2wsdkd_s24 | 2 | 0.002 | 0.3 | 0/F | tf 128/4 | - | 24 | 19670 | 0.38494 | 0.30504 | 0.38500 (19000) | 0.38494 / 0.30504 |
| optim_muwsd_s23 | 2 | 0.001 | 0.3 | 0/F | tf 128/4 | - | 23 | 19670 | 0.38605 | 0.30393 | 0.38605 (19670) | 0.38605 / 0.30393 |
| optim_muwsd_s24 | 2 | 0.001 | 0.3 | 0/F | tf 128/4 | - | 24 | 19670 | 0.38572 | 0.30398 | 0.38572 (19670) | 0.38572 / 0.30398 |
| optim_muwsd_s25 | 2 | 0.001 | 0.3 | 0/F | tf 128/4 | - | 25 | 19000 | 0.38435 | 0.30414 | 0.38435 (19000) | 0.38344 / 0.30390 |
| sx_agedrop_s23 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 23 | 19000 | 0.39328 | 0.31618 | 0.39330 (19670) | 0.39330 / 0.31618 |
| sx_agedrop_s24 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 24 | 19000 | 0.39345 | 0.31543 | 0.39350 (19670) | 0.39350 / 0.31541 |
| sx_base_s23 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 23 | 19670 | 0.39375 | 0.31783 | 0.39375 (19670) | 0.39375 / 0.31783 |
| sx_base_s24 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 24 | 19670 | 0.39288 | 0.31755 | 0.39356 (19000) | 0.39288 / 0.31755 |
| sx_pl20_s23 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 23 | 19670 | 0.39500 | 0.31816 | 0.39500 (19670) | 0.39500 / 0.31816 |
| sx_pl20_s24 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 24 | 19670 | 0.39282 | 0.31764 | 0.39299 (19000) | 0.39282 / 0.31764 |
| sx_pl20_w1_s23 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 23 | 19670 | 0.39463 | 0.31813 | 0.39463 (19670) | 0.39463 / 0.31813 |
| sx_pl20_w1_s24 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 24 | 19670 | 0.39302 | 0.31777 | 0.39302 (19670) | 0.39302 / 0.31777 |
| sx_tau30_s23 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 23 | 19670 | 0.39276 | 0.31674 | 0.39285 (19000) | 0.39276 / 0.31674 |
| sx_tau30_s24 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 24 | 19670 | 0.39206 | 0.31649 | 0.39227 (18000) | 0.39206 / 0.31649 |
| v20_pre2s_e2_s301 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 301 | 19670 | 0.39342 | 0.31754 | 0.39342 (19670) | 0.39342 / 0.31754 |
| v20_pre2s_e2_s302 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 302 | 19670 | 0.39289 | 0.31745 | 0.39289 (19670) | 0.39289 / 0.31745 |
| v20_pre2s_e2_s303 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 303 | 19670 | 0.39461 | 0.31756 | 0.39461 (19670) | 0.39461 / 0.31756 |
| v20_pre2s_e2_s304 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 304 | 19670 | 0.39382 | 0.31807 | 0.39382 (19670) | 0.39382 / 0.31807 |
| v20_pre2s_e2_s305 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 305 | 19670 | 0.39241 | 0.31790 | 0.39262 (19000) | 0.39241 / 0.31790 |
| v20_pre2s_e2_s306 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 306 | 19670 | 0.39191 | 0.31753 | 0.39262 (18000) | 0.39191 / 0.31753 |
| v20_pre2s_e2_s307 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 307 | 19670 | 0.39302 | 0.31778 | 0.39302 (19670) | 0.39302 / 0.31778 |
| v20_pre2s_e2_s308 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 308 | 19670 | 0.39404 | 0.31788 | 0.39404 (19670) | 0.39404 / 0.31788 |
| v20_pre2s_e2_s309 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 309 | 19670 | 0.39308 | 0.31794 | 0.39354 (19000) | 0.39308 / 0.31794 |
| v20_pre2s_e2_s310 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 310 | 19670 | 0.39345 | 0.31757 | 0.39345 (19000) | 0.39345 / 0.31757 |
| v20_pre2s_e2_s311 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 311 | 19670 | 0.39331 | 0.31742 | 0.39331 (19670) | 0.39331 / 0.31742 |
| v20_pre2s_e2_s312 | 2 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 312 | 19670 | 0.39271 | 0.31836 | 0.39310 (18000) | 0.39271 / 0.31836 |
| v12_all_e10 | 10 | 0.002 | 0.3 | 16/T | tf 128/4 | cvnd | 23 | 80000 | 0.39132 | 0.31723 | 0.39270 (77500) | 0.36426 / 0.28929 |
| v12_all_s81 | 10 | 0.002 | 0.3 | 16/T | tf 128/4 | cvnd | 81 | 80000 | 0.39322 | 0.31811 | 0.39322 (80000) | 0.36502 / 0.28963 |
| v12_all_s82 | 10 | 0.002 | 0.3 | 16/T | tf 128/4 | cvnd | 82 | 60000 | 0.39125 | 0.31778 | 0.39292 (70000) | 0.36606 / 0.28898 |
| v12_allprew_s83 | 10 | 0.002 | 0.3 | 16/T | tf 128/4 | cvnd_prew | 83 | 70000 | 0.39266 | 0.31770 | 0.39301 (75000) | 0.36602 / 0.29070 |
| v12_allprew_s84 | 10 | 0.002 | 0.3 | 16/T | tf 128/4 | cvnd_prew | 84 | 80000 | 0.39322 | 0.31793 | 0.39398 (77500) | 0.36725 / 0.29015 |
| v12_allprew_s85 | 10 | 0.002 | 0.3 | 16/T | tf 128/4 | cvnd_prew | 85 | 80000 | 0.39312 | 0.31863 | 0.39326 (72500) | 0.36477 / 0.28925 |
| v13_lr1_mix0 | 10 | 0.001 | 0 | 16/T | tf 128/4 | cvnd | 23 | 85000 | 0.39433 | 0.31845 | 0.39433 (85000) | 0.39361 / 0.31751 |
| v13_lr1_mix1 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd | 23 | 82500 | 0.39389 | 0.31828 | 0.39389 (82500) | 0.38965 / 0.31338 |
| v13_lr1_mix3 | 10 | 0.001 | 0.3 | 16/T | tf 128/4 | cvnd | 23 | 45000 | 0.39149 | 0.31822 | 0.39255 (35000) | 0.36514 / 0.29147 |
| v13_lr2_mix1 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd | 23 | 82500 | 0.39403 | 0.31899 | 0.39455 (87500) | 0.39282 / 0.31605 |
| v14_mix1_s91 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd_prew | 91 | 85000 | 0.39428 | 0.31873 | 0.39428 (85000) | 0.39181 / 0.31584 |
| v14_s92 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd | 92 | 47500 | 0.39158 | 0.31754 | 0.39336 (57500) | 0.38766 / 0.31284 |
| v14_s93 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd | 93 | 82500 | 0.39400 | 0.31822 | 0.39418 (85000) | 0.39053 / 0.31404 |
| v14_s94 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd | 94 | 75000 | 0.39389 | 0.31775 | 0.39404 (62500) | 0.38809 / 0.31348 |
| v14_s95 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd | 95 | 85000 | 0.39249 | 0.31801 | 0.39373 (82500) | 0.38759 / 0.31416 |
| v14_s96 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd | 96 | 47500 | 0.39180 | 0.31788 | 0.39379 (72500) | 0.38868 / 0.31342 |
| v17_pre2s_s101 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 101 | 80000 | 0.39361 | 0.31881 | 0.39530 (87500) | 0.39336 / 0.31598 |
| v17_pre2s_s102 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 102 | 85000 | 0.39532 | 0.31958 | 0.39533 (87500) | 0.39177 / 0.31593 |
| v17_pre2s_s103 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 103 | 87500 | 0.39439 | 0.31860 | 0.39457 (82500) | 0.39318 / 0.31588 |
| v17_pre2s_s104 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 104 | 85000 | 0.39568 | 0.31980 | 0.39593 (82500) | 0.39087 / 0.31713 |
| v17_pre2s_s105 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 105 | 85000 | 0.39416 | 0.31868 | 0.39540 (90000) | 0.39364 / 0.31624 |
| v17_pre2s_s106 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 106 | 85000 | 0.39512 | 0.31927 | 0.39512 (85000) | 0.39287 / 0.31558 |
| v17_pre2s_s107 | 10 | 0.002 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 107 | 82500 | 0.39392 | 0.31900 | 0.39460 (80000) | 0.39254 / 0.31542 |
| v22_pre2s_lr1_s401 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 401 | 50000 | 0.39280 | 0.31794 | 0.39597 (85000) | 0.39087 / 0.31267 |
| v22_pre2s_lr1_s402 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 402 | 75000 | 0.39455 | 0.31851 | 0.39455 (75000) | 0.38991 / 0.31305 |
| v22_pre2s_lr1_s403 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 403 | 77500 | 0.39495 | 0.31906 | 0.39495 (77500) | 0.38853 / 0.31357 |
| v22_pre2s_lr1_s404 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 404 | 85000 | 0.39255 | 0.31846 | 0.39408 (45000) | 0.38952 / 0.31400 |
| v22_pre2s_lr1_s405 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 405 | 67500 | 0.39303 | 0.31824 | 0.39564 (60000) | 0.38826 / 0.31205 |
| v22_pre2s_lr1_s406 | 10 | 0.001 | 0.1 | 16/T | tf 128/4 | cvnd_pre2s | 406 | 85000 | 0.39504 | 0.31847 | 0.39504 (85000) | 0.38958 / 0.31339 |
