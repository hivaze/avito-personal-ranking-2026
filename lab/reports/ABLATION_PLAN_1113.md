# ABLATION PLAN 2026-11-13: component ledger, the launched 2-epoch screen, and the decision tree

Written by the adversarial-review step, 2026-11-13 ~00:25-00:40 UTC (CPU only; nothing under nn/, nn_x/r6, nn_x/w5
touched; see §0.3 for the one state change this step made). Sources: nn_x/abl/ledger/ledger.md (the full ledger,
262 lines), nn_x/abl/{abl_lane.sh, launch_abl.sh, wait_abl_readout.sh, gate_and_launch.sh, abl_eval.py, abl_arms.py,
train.py, tests/test_abl.py}, nn_x/abl/logs/{test_abl_cpu.log, test_abl_gpu.log, gate_and_launch.log}, and the reports
cited per row.

## 0. Review verdict (the five checks) and what was fixed

| check | verdict | evidence |
|---|---|---|
| (1) each arm differs from the base only as intended | **PASS** | `DRY=1 bash nn_x/abl/launch_abl.sh` prints the 26 commands; base == nn_x/w5 lane.sh A_BASE + HYG token for token (parser-level diff in test C0); appended arms end with exactly their flag(s) (`--aux_pos 0`, `--retr_m 32`, `--retr_m 32 --retr_uniq --retr_tcnt`, `--xt_exclude <8 pop>`, `--xt_exclude <10 ctx_nd>`, `--xt_rel_extra <4>`, `--loss_norm batch`, `--blocks 3`, `--no_set`); removal arms drop exactly `--retr_m 16 --attn_gate` / `--sim_bias` / `--domain` from the list (store_true flags cannot be switched off by appending, so the base is a bash array). argparse "last occurrence wins" makes `--aux_pos 0` / `--retr_m 32` / `--loss_norm batch` override the base values |
| (2) flags-off parity proven on the live files | **PASS (re-run)** | CPU: 68/68 (00:12 UTC, logs/test_abl_cpu.log) incl. the AST proof (abl train.py minus exactly the two `add_argument`s, the import, the `--init` guard and the two guarded helper calls == frozen nn_x/w5/train.py, md5 b75830459d99ec1f9a7fea717928e558 = the live w5 train.py; model/cd_arms*/blend/holdout_eval/stale_pred byte-identical; data/common AST-equal up to strings). GPU: the FIRST driver (PID 373813) died at ~00:16:38 WITHOUT a result file (see §0.2); the gate re-ran it under setsid on GPU N at 00:16:44: G1 CD bit-identity 5/5 PASS (identical 39-batch sequence, logits, log lines, checkpoint, val preds; abl flags refused with `--init`), G2 from-scratch identity 5/5 PASS (selected + @last checkpoints, val + holdout preds bit-identical to the frozen w5 train.py), arms-ON checks + stale_pred: see the final line of logs/test_abl_gpu.log and §0.3 |
| (3) readout rule pre-declared and sound | **PASS, one caveat** | Frozen in the abl_eval.py docstring at 00:08 UTC, before any abl run (the only GPU jobs before it were the harness tests). Paired per feed on the same seed, pooled over seeds 23/24, SE_feed = sd/sqrt(n_feeds), SE_cal = sqrt(SE_feed² + σ_run²/2) with the r7 A/A σ_run (0.00035 val / 0.00015 holdout; same convention as w5_eval.py); slices M<50, M>=50, Tue, Thu, coldpos, warmpos, holdout all + late20%; primary = LAST checkpoint (holdout unselected), selected = info. Selftest on r7_dwa vs r7_base reproduces R7_ARCH_HPARAM's numbers (+0.00064 z_cal +1.32 / -0.00027). **Caveat:** the SCALE-UP gate is `PASS iff val > +1 SE_cal AND holdout >= 0`, which is looser than the r7 stage-1 rule (1.5 SE_cal + both seeds); the r7 rule is computed as the info flag STRICT-PROMOTE. The decision tree below uses STRICT-PROMOTE to prioritise the 10-epoch pairs and treats a bare PASS as "pair only if a GPU is free". The rule itself is left as frozen (changing it after the freeze would defeat the pre-declaration) |
| (4) ledger numbers are real | **PASS** (12 spot-checked, all exact) | A4 +0.00186 / +0.00326, A1+A4 +0.00273 / +0.00389, 3-seed +0.00278 / +0.00382, Muon +0.00201 / +0.00295, Muon+WSD 3-seed +0.00306 / +0.00331 (nnx_VERDICT.md rows 1-3, §4 table); nohist 0.36808 vs base 0.37361, noset 0.37386, nopos_sim 0.37624 vs nopos 0.37385, v2_batch 0.37804 / 0.29619 vs v2_feed 0.38003 / 0.29676 (+0.0020 / +0.0006) vs v2_k9 0.38039 / 0.29892 (+0.0024 / +0.0027) (nn_report.md ablation + v2 tables); hl2 -0.00059 / +0.00007 (M>=50 -3.27), dwa +0.00064 / -0.00027, SE_cal 0.00048-0.00061 (R7_ARCH_HPARAM §3); m02 vs c0 +0.00117 ± 0.00039 (z 2.97) / +0.00134 ± 0.00022 (z 5.98) (CURRENT_STATUS 11-12 line 142); tau30 -0.00091 ± 0.00049 / -0.00109 ± 0.00028 (z -3.86) WORSE (sol_gpu.md row A); popularity replay -0.0057 / -0.0151, ctx_nd -0.0110 / -0.0070, st_pop -0.00124 / -0.00145 (findings.txt §3-4, lines 49-50, 72, 168-169, 195); a4u 2-ep +0.00117 ± 0.00047 (z 2.48), 10-ep +0.00036 ± 0.00068 / +0.00018 (CURRENT_STATUS line 115; EXPERIMENT_REGISTRY 12:37); G3 0.40032 / 0.32379, F7x 0.39667 / 0.31999 (EXPERIMENT_REGISTRY); w5_e10_base_s951-954 val 0.39632 / 0.39691 / 0.39745 / 0.39549 (their .json); r7_base_s23/24 0.39408 / 0.39334, 0.31779 / 0.31749 (r7 eval_screen.txt). The one hist-512 claim (raw cap 256, 28.8% at cap) was verified by the ledger step on the raw parquet and is consistent with data/flat (max H 256) |
| (5) GPU rules | **PASS, one environment change** | abl_lane.sh accepts only GPUs N (refuses <ids>, tested), `CUDA_VISIBLE_DEVICES` = one id, 12 CPU threads, nvidia-smi before EVERY launch under a per-GPU flock (<= 3 compute processes = "< 4", >= 60 GB free), the lock held until the job has built its model and loaded its data (so the next check counts it), at most 3 lanes per GPU; train.py PIDs verified through /proc/<pid>/cmdline + pgrep and written to nn_x/abl/RUNNING.txt; the waiter's stale_pred and the gate's re-run use the same check. **GPUs N are shared with another workstream**: at 00:17 another job (PID [n], `nn_x/w25/train_w25.py ... --tag A1_base_xm`, 69 GB, started 00:08:57) sat on GPU N; it finished before our first launch (lane 5#1 saw 0 processes at 00:24:27). At 00:28:51 two new w25 jobs started, one per GPU (PIDs [n] / [n], 27-31 GB), so each GPU now carries 3 of ours + 1 other = 4 processes and ~[n] GB of (GPU memory size removed). Our launches all happened at <= 2 compute processes (the rule's "< 4"), at most 3 of ours per GPU, nothing foreign touched. Next launches will see 3 processes and ~[n] GB free (admitted); if a second other job lands on a GPU, the lane check (n <= 3) automatically throttles us to 2 per GPU (polls every 120 s, up to 6 h) |

### 0.1 Issues found (and their status)

1. **The first GPU test driver was killed, not failed.** Evidence: logs/w5_ablt_fs_ref.log stops after `save_last`
   (the "@last" checkpoint / holdout preds exist, the selected-checkpoint holdout file and its "saved holdout preds" log
   line do not), no scratch stdout for that run was written (e2e_run writes it right after the subprocess returns), no
   logs/test_abl_gpu.json, and the gate saw "rc 2 (missing)" at 00:16:43, ~1 min after the previous step's hand-off.
   The driver had been started from the previous step's shell without setsid, so it went down with that shell; the
   gate (setsid) survived and executed its one-shot re-run under setsid. No code change was needed; the re-run passed
   the same checks on the same files. Lesson for the next launches: start every long GPU driver with
   `setsid nohup ... < /dev/null &` (abl_lane.sh / launch_abl.sh already do).
2. **Scale-up PASS bar** (see check 3): kept as frozen, handled in the decision tree.
3. **Claim directories persist after a failure**: abl_lane.sh claims `logs/queue/claim_<v>` and on rc != 0 touches
   `logs/queue/failed_<v>` and never retries. To re-run a failed job: `rmdir nn_x/abl/logs/queue/claim_<v>; rm
   nn_x/abl/logs/queue/failed_<v>` and start one lane with that job (`setsid nohup bash nn_x/abl/abl_lane.sh <5|7>
   "<arm>:<seed>" > nn_x/abl/logs/lane_abl_rerun.out 2>&1 < /dev/null &`). A failed BASE seed blocks every arm of that
   seed in the readout (abl_eval prints "incomplete"); re-run it first.
4. **Screen / recipe mismatch to keep in mind when reading**: the 2-epoch base runs at lr 1e-3 (the 10-epoch recipe
   uses 2e-3, worth +0.0008 at 10 ep) and at mix 0.02 it sees 0.19 exploration-pool passes (0.65 at 10 ep). Ablations
   of things that act in the cooldown (K9 anneal, domain token on pool feeds) are under-dosed at 2 ep; the verdicts
   below therefore always route a recipe change through a 10-epoch pair.
5. The ledger's §4 says the w5 lane's ALLOWED default is "<ids>": confirmed (nn_x/w5/lane.sh line 34); abl_lane.sh
   uses `ABL_GPUS` default "5 7".

### 0.2 Timeline of the harness (UTC, 11-13)

23:52 ledger written -> 00:03 nn_x/abl copied from nn_x/w5 (md5s in tests/w5_source_md5.txt) -> 00:04 abl_arms.py +
train.py flags -> 00:07 lane / launch / waiter -> 00:08 readout rules frozen (abl_eval.py docstring) -> 00:12 CPU
tests 68/68 -> 00:12:45 GPU test driver 1 (PID 373813) -> 00:15:18 gate v2 (PID 388525) -> ~00:16:38 driver 1 died
without a result -> 00:16:44 gate re-ran the GPU test on GPU N (driver PID 393576) -> G1 5/5 PASS 00:19, G2 flags-off
5/5 PASS 00:22 -> (arms-ON + stale_pred) -> gate writes logs/FROZEN_abl.txt and starts launch_abl.sh on a clean pass.

### 0.3 Launch record

- 00:24:26 UTC: GPU test re-run **17/17 PASS** (logs/test_abl_gpu.{log,json}, `failed: []`, 462 s): G1 CD identity 4/4 +
  init refusal; G2 from-scratch identity 4/4 + hygiene check; xdrop_pop / xrel_more arms-ON: checkpoint cfg layouts ==
  the CPU C1 layouts (26 cols / xt_rel pre2-only / n_dense 92; 34 cols / xt_rel 9 / n_dense 112), same batch sequence
  as flags off with a different checkpoint, setup log lines present; stale_pred.py writes both stale-val files and uses
  only the pre2 overrides for the pop-free checkpoint. Freeze record logs/FROZEN_abl.txt (md5 of every harness file:
  train.py 7c8d4d1a0ad1291687a0578a1e664adc, abl_arms.py 4e1e0de0c9431a95edf29de5bdf3ba3e, abl_eval.py
  541174fd4c1967a4de7aae709b2f8479, abl_lane.sh 93448fde8bcd34de878433ef36d71046, ref_train_w5.py.txt
  b75830459d99ec1f9a7fea717928e558 = live nn_x/w5/train.py).
- 00:24:26 the gate started launch_abl.sh (PID 412503); lanes (bash abl_lane.sh, setsid) and their first jobs, every
  train.py PID verified through /proc/<pid>/cmdline and pgrep (nn_x/abl/RUNNING.txt):

  | lane PID | GPU | first job | train.py PID | start (UTC) | GPU state at the check |
  |---|---|---|---|---|---|
  | 412508 | N | abl_base_s23 | 412531 | 00:24:29 | 0 compute processes, [n] GB free |
  | 413566 | N | abl_base_s24 | 413586 | 00:25:29 | 0, [n] GB |
  | 415817 | N | abl_no_aux_s23 | 415843 | 00:26:29 | 1, [n] GB |
  | 417576 | N | abl_no_aux_s24 | 417600 | 00:27:29 | 1, [n] GB |
  | 418806 | N | abl_no_retr_s23 | 418833 | 00:28:29 | 2, [n] GB |
  | 421396 | N | abl_no_retr_s24 | 421425 | 00:29:29 | 2, [n] GB |
  | 424048 | - | waiter wait_abl_readout.sh (polls 120 s; readout -> stale_pred -> final readout) | - | 00:30:26 | - |

- GPU occupancy at 00:30:56: GPU N [n] GB (ours 57.2 / 57.1 / 55.6 GB + another w25 job PID 419241, 26.5 GB, started
  00:28:51), GPU N [n] GB (ours 57.1 / 57.1 / 55.5 GB + other w25 job PID 419236, 31.3 GB, started 00:28:51); both at
  95-96 % utilisation. The first other job on GPU N (PID 368155, 69 GB) had finished before our first launch (the
  lane saw 0 processes). The w25 jobs belong to another workstream and were not touched; when a lane launches its next
  job it will see 3 processes (2 ours + 1 other) and ~[n] GB free, which the rule admits.
- Health at 00:31: abl_base_s23 step 5000/19670 val 0.38685 / holdout 0.31059 (4.7 min); abl_base_s24 step 5000 val
  0.38698 / holdout 0.31018; abl_no_aux_s23 step 2000 val 0.38048 (loss 2.20 = main loss only, as expected without the
  aux term); CLIPLOG plateau clipped-share 0.000. Step rate under 4 processes per GPU ~1.4 min / 1000 steps -> ~28-32
  min per run (+ ~2 min for the @last / holdout preds) -> 26 runs over 6 lanes ≈ 4.3 rounds -> all runs done ~02:50-03:15
  UTC, abl_eval.txt right after, stale preds (+~15 min on one GPU) and the final abl_eval.txt by ~03:30-03:45 UTC.
- State changes made by this review step: none to the harness or the rules (the gate's re-run was its own, pre-written
  behaviour); the only files written are this report and the scratch outputs of the checks. No process was stopped.

## 1. The component ledger (condensed; full rows with sources in nn_x/abl/ledger/ledger.md §1)

Δ = paired per-feed delta ON minus OFF (val / holdout), i.e. the component's measured worth. Recipe: CURRENT = the live
Muon+WSD+retr16+pre2s lineage (or its 2-ep screen form), CURRENT-ish = same lineage with one setting different (e.g. mix
0.1), OLDER = AdamW-cosine era before retrieval / pre2s / Muon. "Current?" = was it ever measured on the current recipe.

| # | component (current setting) | measured worth | horizon / seeds / recipe | source | current? | gap | screen arm |
|---|---|---|---|---|---|---|---|
| A1 | history encoder (1 pre-LN layer over <= 256 events) | +0.0055 (nohist 0.36808 vs 0.37361) | 2 ep, 1 seed, OLDER | nn_report ablations | never | 3 generations stale; nobody would remove it | none (`--no_hist` exists) |
| A2 | hist_layers 1 | hl2 -0.00059 / +0.00007, M>=50 -3.3 SE | 2 ep, s23/24, CURRENT-ish | R7 §3 | yes | closed | - |
| A3 | arch tf (GRU variant) | GRU ≈ tf (+0.0003..+0.0008 curve-matched), no pool gain | 10 ep, 6 seeds, CURRENT-m02 | EXPERIMENT_REGISTRY | yes | closed (family exists) | - |
| A5 | max_hist 256 | inference crop 64/128/192/256: 0.3754/0.3782/0.3794/0.3799 (OLDER); crop TTA -0.0010 (CURRENT) | OLDER + CURRENT | nn_report K10; R6_PLAN §3 | partly | **hist 512 infeasible**: raw history capped at 256 by the organisers (max 256, 28.8% at cap, median 187) | - |
| B1 | candidate-set self-attention | neutral (noset 0.37386 vs 0.37361) | 2 ep, 1 seed, OLDER | nn_report | never | stale: 34 within-feed-relative inputs exist now; cset (prior + cand. retrieval) 2 ep +0.0003/+0.0009, 10 ep +0.0002/-0.0000 | **no_set** (cost ~0.75x) |
| B2 | blocks 2 | blocks 3 -0.0006 UNPAIRED (AdamW, no xt); "deep" ±0.0002 (AdamW cvnd) | OLDER | nn_report; r6_recap §3.5 | never under Muon | stale -> re-test qualifies; low prior (r7 block arms cost M>=50, d192 memorises) | **blocks3** (cost ~1.3x) |
| B3 | d 128 / 4 heads | heads 8 -0.0011; d256 -0.0008; **d192 (F12, Muon) 0.39371/0.31843 vs F7 0.39460/0.31911** | OLDER + CURRENT | nn_report; r6_recap | yes | closed | - |
| B5 | block form (pre-LN, GELU FFN, plain residuals) | r7 0/13: derf -0.0012, periln -0.0011, batch2s -0.0015, muon_in -0.0010, swiglu -0.0011, ftok -0.0011, diff2 -0.0009, hl2 -0.0006, pbld -0.0006, postln -0.0005, hc4 -0.0002, resdrop0 +0.0003, dwa +0.0006 (holdout -0.0003) | 2 ep s23/24 CURRENT-ish | R7 §3 | yes | closed at screen power | - |
| C1 | retr_m 16 (TabR-style top-16 history items per candidate) | A4 +0.00186 / +0.00326; A1+A4 +0.00273 / +0.00389 (3-seed +0.00278 / +0.00382); 10 ep cosine +0.0028 / +0.0030; T(k_c-k_j) carries ~90% | 2 ep s23/24/25 OLDER; 10 ep OLDER | nnx_VERDICT §2.3; r6_recap | **never removed on current** | overlap with pre2s / ctx_nd unknown; retr_m 8/32/64 alone NEVER (a4w confounded by the product term); a4u +0.00117 (2 ep) -> +0.00036 (10 ep), warm -0.0034 / cold +0.0095 | **no_retr**, **retr32**, **retr32u** |
| C2 | attn_gate | +0.00042 alone n.s.; +0.00087 over A4; mean of 4 pairs +0.00065 ± 0.0003 | 2 ep OLDER | nnx_VERDICT | never removed | small, cheap | removed together with retr in no_retr |
| C3 | sim_bias (raw cosine added to cross-attention logits, per-head scale) | +0.0024 (0.37624 vs 0.37385) | 2 ep, 1 seed, OLDER | nn_report | **never removed** | retrieval now has its own a·S term | **no_sim** |
| D1 | user0 (K5 dense flags) | neutral (unpaired, with K2) | 2 ep OLDER | nn_report v2 | never removed | `--xt requires --user0` -> needs code; low value | - |
| D2 | domain token (K2: production / random-like / exploration; M<50 hard-labelled production) | neutral (unpaired) | 2 ep OLDER | nn_report v2; INSIGHTS 3.1 | **never removed on current** | DOMAIN-DIAG never run; L3 / L6 gated on it; TRUNC readout credits "K9 + domain" for absorbing the production artefact | **no_domain** |
| D3 | no_pos (position tower off) | tower -0.0002; TRUNC-PROD -0.00048 ± 0.00051 / -0.00009 FAIL (4 x 10 ep, CURRENT) | OLDER + CURRENT | nn_report; CURRENT_STATUS 22:54 | yes | closed | - |
| E1 | xt preset cvnd_pre2s (34 inputs: pop 8, ctx_nd 10, ctx_hn 6, ctx_hs 4, pre2 6) | cv +0.0032 / +0.0092; cvnd over cvraw +0.0000 / +0.0014; pre2s +0.00191 ± 0.00058 (3 ep) and +0.0009 / +0.0005 (10 ep); **replay reliance (CURRENT members)**: pop -0.0057 / -0.0151, ctx_nd -0.0110 / -0.0070, ctx_hs -0.0032 / -0.0050, ctx_hn -0.0015 / -0.0014, pre2 -0.0019 / -0.0046; private-like cost st_pop -0.00124 / -0.00145, pool -0.00175 (pmix -0.00227) | OLDER (retrain pairs); CURRENT (replays) | nn_report; r6_recap §3.4; findings §3-4 | pre2 yes; **pop / ctx_nd / ctx_hn / ctx_hs retrain-without NEVER** | replay != retrain (old ctx pair: +0.0000 retrain vs -0.011 replay); popularity is the only private-shifted family | **xdrop_pop**, **xdrop_ctxnd** (new flag `--xt_exclude`) |
| E2 | xt_rel (in-model within-feed pct-rank + z for 5 columns) | FR-1 rank channel on pre2 +0.0007 ± 0.0005 (3 ep) | OLDER-ish | nn_report 18:15; FEATURES_REVIEW §2.1 | never | never on more inputs; ranks are split-invariant (private-robust by construction) | **xrel_more** (new flag `--xt_rel_extra`) |
| E3 | f1..f5 (organisers' production scores, 15 dense cols + wide term) | never ablated / replayed; f1 strongest single input; f1/f3/f5 slopes 1.14-1.23x stronger on production feeds | - | findings §1c | never | reliance unknown; read by zero / permute REPLAY on an existing checkpoint (no training) | not in the screen |
| F1 | listwise softmax CE | BCE -0.0009; 14 proxy alternatives none better; CE's gradient allocation ≈ LambdaRank | OLDER; PROXY; CURRENT analysis | nn_report; sol_solvers; INSIGHTS 6 | yes | closed | - |
| F2 | loss_norm feed (K1b) | +0.0020 / +0.0006 (v2_feed vs v2_batch) | 2 ep OLDER, 1 pair | nn_report v2 | **never re-measured** | aux is still batch-normalised (AUXF null) | **lnorm_batch** |
| F3 | K9 aux (listwise CE to softmax(-pos/10) on production feeds, w 0.5 (1-p), annealed 0.1) | MSE aux vs none +0.0020; listwise vs MSE +0.0024 / +0.0027; variants: tau30 -0.00091 / -0.00109 WORSE, PL20 +0.00059 null, PL+w1.0 +0.00051 null, aa5 STOP, auxf STOP, intent head -0.0014 | OLDER (worth); CURRENT-ish (variants) | nn_report v2; sol_gpu; CURRENT_STATUS | variants yes; **aux_pos 0 NEVER with retrieval / pre2s / Muon** | the "trunk regulariser"; credited for absorbing the production artefact | **no_aux** |
| G1 | Muon | +0.00201 / +0.00295 (2 ep); +0.0012 / +0.0010 (10 ep cosine); muon_in -0.0010 | OLDER; CURRENT-ish | nnx_VERDICT; R7 | partly | adopted; no reversal signal | - |
| G2 | WSD (1-sqrt cooldown, no stage 2) | +0.00114 on AdamW, +0.00158 on Muon; stage 2 after WSD -0.0012 / -0.0015; cooled 7-seed ensemble vs plateau +0.0031 ± 0.0006 | OLDER + CURRENT | nnx_VERDICT; r6_recap §5 | yes | adopted | - |
| G3 | lr 2e-3 at 10 ep / 1e-3 at 2 ep | 10 ep 2e-3 vs 1e-3 +0.00078 / +0.00067; 2 ep 2e-3 -0.00097 val | CURRENT | r6_recap §3.1 | yes | horizon-dependent (screen mismatch) | - |
| G4 | cooldown_frac 0.2 | cf4 -0.00121 / -0.00023 (coldpos -0.0058), e3 -0.0001, s2c2 -0.0001: STOP | CURRENT (CD) | CURRENT_STATUS 15:29 | yes (longer) | shorter never; low prior | - |
| G5 | cooldown_mix 0.02 (m02) | +0.00117 ± 0.00039 / +0.00134 ± 0.00022 vs mix 0.1, all slices positive, 7-member pool +0.00108 / +0.00137; m0 ≈ m02 on val (coldpos -0.0023); 0/15 further cooldown arms | CURRENT (CD re-cools) | CURRENT_STATUS 10:55-22:54 | yes | dose 0.01 / 0.04 never; **2-ep screen is blind** (0.19 pool passes) -> CD harness (stage B, roots 101-104 vs w5_cd_m02ref) | not in the screen |
| G6-G9 | aux_anneal 0.1, wd 0.02, dropout 0.1, batch 32768/512 | aa5 STOP; w10 STOP; d0 -0.0012, d20 -0.0002, resdrop0 +0.0003, adrop0 FAIL; batch2s -0.00145 / -0.00109 | CURRENT / CURRENT-ish | R7; CURRENT_STATUS | yes | closed | - |
| G10 | epochs 10 | F9 (10 ep) vs F8 (2 ep): single +0.0006 / +0.0007, 6-seed ensemble +0.0040 / +0.0031; 12 ep -0.0001 | CURRENT | r6_recap §3.2 | yes | none | - |
| H1 | 30K random-like holdout + `--select_on holdout` + select_min_step 12k + last ckpt kept | the only val-label-free selector; hygiene produced the best single holdouts (0.3200-0.3208); same-seed nondeterminism |Δval| ~0.0007 | CURRENT | nnx_VERDICT §2.2; CURRENT_STATUS 22:54 | yes | protocol, not a lever | - |
| I1-I3 | val FT n2cpv3, blend 0.5/0.5, member-equal z-mean pool | FT ≈ 0 at pool level on pre2s/m02 trunks; FT half worth ~+0.0005 public; pool saturated (headroom ~+0.0002) | CURRENT (public) | INSIGHTS 1, 10; CURRENT_STATUS 21:52 | yes | only stronger members move the pool | - |

**Reading.** (i) Every large measured worth (history +0.0055, retrieval +0.002-0.004, sim_bias +0.0024, K9 +0.002 /
+0.0024, loss_norm +0.0020, cv inputs +0.003 / +0.009) was measured on the OLDER recipe, at 2 epochs, mostly on one seed;
the components overlap (retrieval and sim_bias both read raw cosine; pre2s and ctx_nd both encode history similarity; K9
and the domain token both encode the production regime), so those numbers are upper bounds for the current worth.
(ii) Nothing in the current recipe has ever been removed on the current recipe. (iii) The open scale-up space is narrow:
retrieval depth / dedup, rank channels on more inputs, blocks 3 under Muon, domain-token variants, m02 dose (CD harness).
hist 512 is infeasible (data cap).

## 2. The launched screen (nn_x/abl, 2 base + 12 arms x seeds 23 / 24 = 26 two-epoch runs)

**Base** (== nn_x/w5 lane.sh A_BASE + HYG; the wave-5 stage-A base, which had never been run), variant abl_base_s<23|24>:
```
cd <repo>/nn_x/abl && CUDA_VISIBLE_DEVICES=<5|7> OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 <repo>/.venv/bin/python train.py --variant abl_base_s<seed> \
 --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout \
 --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.02 \
 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 1000 --no_test \
 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed <23|24>
```
19,670 steps (9,835 / epoch), cooldown from step 15,736; selected = holdout-best eval at step >= 12,000; LAST = step 19,670.

| rank | arm | type | exact change vs base | expected 2-ep val Δ (arm - base) | what it decides |
|---|---|---|---|---|---|
| N | no_aux | ABLATION | `+ --aux_pos 0` (aux head not built) | -0.002..-0.005 | worth of K9 on the current recipe; whether aux work deserves GPU; the TRUNC-FAIL story |
| N | no_retr | ABLATION | `- --retr_m 16 --attn_gate` | -0.0015..-0.003 | worth of retrieval + gate on current inputs; bounds the retrieval scale-ups |
| N | retr32 | SCALE-UP | `+ --retr_m 32` | -0.0003..+0.0008 (cost ~1.2x, +2-3 GB) | retrieval depth alone |
| N | retr32u | SCALE-UP | `+ --retr_m 32 --retr_uniq --retr_tcnt` | 0..+0.0012 (cost ~1.3x) | depth over distinct items; retr32u - retr32 isolates the dedup; expect a warm/cold trade |
| N | xdrop_pop | ABLATION | `+ --xt_exclude seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo` (26 xt cols left, n_dense 104 -> 92) | -0.002..-0.005; stale-val Δ ≈ val Δ + 0.0012 | worth of the only private-shifted family; PRIVATE TRADE = stale-val Δ - val Δ |
| N | xdrop_ctxnd | ABLATION | `+ --xt_exclude ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_h,ctx_nd_knn_h_dev,ctx_nd_knn_u,ctx_nd_knn_u_dev,ctx_nd_max,ctx_nd_top5h,ctx_nd_top5u,ctx_nd_ucent` (24 cols, n_dense 94) | -0.0005..-0.003 | whether the most-relied-on family is real worth or redundant with retrieval / set attention |
| N | no_sim | ABLATION | `- --sim_bias` | -0.0005..-0.0025 | whether the cosine prior is redundant with the retrieval's a·S |
| 8 | no_domain | ABLATION | `- --domain` (p_random still loads via `--holdout`; pool and K9 (1-p) unchanged) | -0.001..+0.0005 | worth of the domain token; gates INSIGHTS L3 / L6 |
| 9 | xrel_more | SCALE-UP | `+ --xt_rel_extra seq_cv_pop,it_hist_et1_luo,ctx_nd_knn_h,ctx_nd_ucent` (xt_rel 5 -> 9, n_dense 112) | -0.0003..+0.0008 | rank channels for the 4 most-relied-on rank-free inputs; split-invariant -> private lever (read stale-val) |
| 10 | lnorm_batch | ABLATION | `+ --loss_norm batch` | -0.001..-0.0025 | worth of K1 (equal per-feed weight); read M<50 vs M>=50 |
| 11 | blocks3 | SCALE-UP | `+ --blocks 3` | -0.0005..+0.0005 (cost ~1.3x) | the only capacity knob with stale evidence |
| 12 | no_set | ABLATION | `+ --no_set` | -0.001..+0.0003 (cost ~0.75x) | worth of the set attention; if neutral, a ~25% step-time lever |

Job order (shared list, atomic mkdir claims, 6 lanes = 3 per GPU on GPUs N, staggered 60 s): base:23 base:24
no_aux:23 no_aux:24 no_retr:.. retr32 retr32u xdrop_pop xdrop_ctxnd no_sim no_domain xrel_more lnorm_batch blocks3
no_set (each x 23 / 24). Every run writes logs/<v>.{log,json,stdout}, ckpt/<v>.pt + <v>@last.pt, preds/nn_<v>[@last]_trainonly_val.npy,
preds/holdout/nn_<v>[@last]_holdout.npy. Measured wall: ~1.4 min / 1000 steps with 4 processes per GPU -> ~30 min per
run (blocks3 / retr32u ~1.3x), 26 runs / 6 lanes -> all done ~02:50-03:15 UTC; the waiter (wait_abl_readout.sh) then
runs abl_eval.py (CPU, ~3 min) -> logs/abl_eval.txt (+ .json; copy abl_eval_prestale.txt), stale_pred.py for base /
xdrop_pop / xdrop_ctxnd / xrel_more (selected + @last checkpoints, one GPU, ~15 min), and the final logs/abl_eval.txt
with the STALE-VAL block (~03:30-03:45 UTC). Progress: logs/wait_abl_readout.log,
logs/lane_abl_gpu{5,7}_{1,2,3}.out, nn_x/abl/RUNNING.txt (START / END lines with verified PIDs).

**Pre-declared readout (abl_eval.py docstring, frozen 00:08 UTC; unchanged by this review):**
- Primary = LAST checkpoint (k 19,670); selected = info. Δ = arm - base, same seed, per feed, pooled over seeds 23/24;
  SE_feed = sd/sqrt(n); SE_cal = sqrt(SE_feed² + σ_run²/2), σ_run 0.00035 val / 0.00015 holdout. Slices: val M<50,
  M>=50, Tue, Thu, coldpos, warmpos; holdout all, late20%.
- ABLATION (no_aux, no_retr, xdrop_pop, xdrop_ctxnd, no_sim, no_domain, lnorm_batch, no_set): worth = -Δ.
  LOAD-BEARING iff val Δ < -2 SE_cal AND holdout Δ < 0. DISPENSABLE iff |val Δ| <= 1 SE_cal AND holdout Δ >= -1 SE_cal
  AND no val slice < -2 SE_feed (confirm at 10 ep before adopting). Else INCONCLUSIVE.
- SCALE-UP (retr32, retr32u, xrel_more, blocks3): PASS iff val Δ > +1 SE_cal AND holdout Δ >= 0 -> 10-epoch pair.
  Info flags: STRICT-PROMOTE (r7 stage-1: val > 1.5 SE_cal, holdout >= 0, both seeds positive on val or on holdout);
  COLD/WARM TRADE (|coldpos - warmpos| > 2 sqrt(SE_c² + SE_w²) -> stale-val + pmix required before adoption).
- Diagnostics (never gates): per-seed Δ, selected read, pmix (private-mix pooled Δ), 08_levers per-positive-class loss
  table, CLIPLOG, train-minutes ratio vs base, and for xdrop_pop / xdrop_ctxnd / xrel_more the STALE-VAL Δ
  ('both_pre2both' private-like inputs) with PRIVATE TRADE = stale-val Δ - val Δ.
- Expected SE_cal at this screen size: ~0.0005-0.0006 val, ~0.00025-0.00035 holdout (R7 §3), so LOAD-BEARING needs a
  worth of >= ~0.0011 and a 2-ep ablation effect of 0.002+ (the ledger's expectations for ranks 1, 2, 5) will be
  unambiguous; DISPENSABLE needs |Δ| <= ~0.0006.

## 3. Decision tree for tomorrow (read logs/abl_eval.txt; the SUMMARY table has one line per arm)

### 3.0 Sanity before reading any verdict
1. `== runs` block: all 26 present, selected step >= 12,000, LAST = 19,670, 0 "(missing)"; base val expected ~0.393-0.395
   (r7_base at mix 0.1 was 0.39408 / 0.39334; m02 at 2 ep is a small change), holdout ~0.317-0.318. A base more than
   ~0.002 below that or a seed pair differing by > 0.0015 -> suspect the GPU-5 crowding (4 processes) rather than the
   arm; the MINUTES line shows it (the ratio vs base is a cost read, and base runs themselves may be slow).
2. CLIPLOG block: cooldown clipped-share ~0 as in w5; a run with NaN / clipped-share 1.0 is a crash to re-run (§0.1 item 3).
3. If abl_eval.py itself failed (logs/wait_abl_readout.log "readout exit != 0"): it is a CPU script, fix and re-run
   `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/abl_eval.py screen`; the rules are not touched.
4. If a run failed: §0.1 item 3 (rmdir the claim, rm the failed marker, re-run that one job on GPU N with the rules);
   a failed base seed first.

### 3.1 Scale-ups (retr32, retr32u, xrel_more, blocks3): which go to 10-epoch pairs

Priority order for the 10-epoch pairs (each pair = 2 runs x ~3 h on one GPU each; at most two pairs can start tomorrow
morning on GPUs N next to whatever else runs):
1. **STRICT-PROMOTE** (val > 1.5 SE_cal, holdout >= 0, both seeds agree): run the pair immediately.
2. **PASS without STRICT** (1 SE_cal < val <= 1.5 SE_cal or seeds disagree): run the pair only if a GPU is free after
   (1); otherwise add seed 25 to the 2-ep screen first (one base + one arm run, 2 x 40 min; the readout pools 3 seeds
   automatically if `SEEDS` in abl_eval.py is extended — a diagnostic re-run, the rule is unchanged).
3. **FAIL**: closed; record the number in the ledger / registry.
4. **COLD/WARM TRADE flagged** (expected for retr32u, the a4u pattern): the pair is still run if it promotes, but adoption
   additionally needs stale-val Δ >= 0 and pmix Δ >= -1 SE at 10 ep (a4u lost warm positives -0.0034 for cold +0.0095
   and its 2-member mixed ensemble was no better than 2x base).

10-epoch pair command (the wave-5 stage-E base + HYG, seeds 951 / 952; the references w5_e10_base_s951 / s952 are exact
flags-off references because nn_x/abl train.py is proven bit-identical to nn_x/w5 train.py at flags off, and the w5
chain of proofs (test_w5 / trunc_lifts / hzcd / valcd, all `failed: []`) connects the live w5 train.py to the 3c7667e
version those runs used):
```
cd <repo>/nn_x/abl && CUDA_VISIBLE_DEVICES=<5|7> OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 \
NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 setsid nohup <repo>/.venv/bin/python train.py --variant abl_e10_<arm>_s<951|952> \
 --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout \
 --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2 \
 --aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500 \
 --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed <951|952> <arm flags, appended last> \
 > logs/abl_e10_<arm>_s<seed>.stdout 2>&1 < /dev/null &
```
(nvidia-smi check first: < 4 compute processes and >= 60 GB free on that GPU; record the train.py PID in
nn_x/abl/RUNNING.txt after a /proc/<pid>/cmdline check. abl_lane.sh refuses seeds other than 23 / 24, so the pairs are
launched by hand or with a 3-line wrapper; they take ~3 h each on an idle GPU, 180 min for the w5 e10 bases.)
Readout of a pair: paired per feed vs w5_e10_base_s951 / s952 with the w5_eval.py machinery (`R.DIRS["abl"]` is registered
by abl_eval.py; a 6-line `pairs` list is all that is needed), unified rule **val > 2 SE, holdout >= 0, no slice < -2 SE**,
plus the ensemble-level check: z-mean the two arm members into a copy of the G3-type pool in place of two base members
and require the pool val / holdout not to drop (the pool is saturated: only a stronger member moves it).

What a positive result means per arm:
- **retr32** promotes -> retrieval depth is live: run **retr64** next (2-ep screen, +4-6 GB, ~1.5x) and the 10-ep pair of
  the better depth; if retr32u also promotes, compare retr32u - retr32 (the dedup's own worth at depth 32).
- **retr32u** promotes but retr32 does not -> the dedup / type-count mechanism, not the depth, carries it (a4u again):
  the 10-ep pair is what decides (a4u's 2-ep +0.0012 shrank to +0.0004 at 10 ep).
- **xrel_more** promotes -> in-model rank channels are a cheap lever: (a) 10-ep pair, (b) read STALE-VAL: a PRIVATE
  TRADE > 0 (the arm loses less under the private-like coverage shift) makes it a private lever even at a small val
  gain, (c) second-wave arm: rank/z for the ctx_hs_* and ctx_hn_* columns too.
- **blocks3** promotes -> re-test depth with hl2 removed from the question (r7's hl2 cost M>=50); if the 10-ep pair
  confirms, it is a cost (1.3x) vs gain trade for the next member families, not a change to the existing pool.

### 3.2 Ablations: which would change the recipe

**LOAD-BEARING** (val Δ < -2 SE_cal, holdout < 0): the component stays; the information is what it unlocks:
- **no_aux** load-bearing (expected): K9 is worth real NDCG on the current recipe -> aux-side work still has a lever
  (the never-run forms are `--aux_pos 1.0` listwise alone and `--aux_anneal 0`; both horizon-dependent -> CD harness),
  and the TRUNC-FAIL explanation ("K9 + domain absorb the production artefact") keeps its support. Read M<50 vs M>=50:
  K9 acts on production feeds, M<50 feeds are hard-labelled production.
- **no_retr** load-bearing: its worth bounds what ranks 3-4 can add. If the worth is still >= +0.002, retrieval is the
  component to scale (retr64, retr_cat = same-category retrieval, dk 128: second-wave code). If it has shrunk to
  ~+0.001, pre2s / ctx_nd already carry most of it and retrieval scale-ups are closed unless rank 3/4 promoted.
- **no_sim** load-bearing: attention priors are not saturated -> revisit cset-type priors at 10 ep (cset 2 ep +0.0003 /
  +0.0009, 10 ep +0.0002 / 0.0000) only if a prior-type arm promotes somewhere else; otherwise record.
- **no_domain** load-bearing: the domain token is real -> INSIGHTS L6 (soft p for M<50) and L3 become second-wave arms
  (new code, ~20 lines) and DOMAIN-DIAG (gpu_domain_diag.py, written, never run) is worth its GPU minutes.
- **xdrop_pop** load-bearing (expected): popularity inputs buy val. Then read the PRIVATE TRADE: if stale-val Δ is
  >= -1 SE while val Δ <= -2 SE, the val worth of popularity does not survive the private-like coverage shift -> a
  **pop-free member family** is a diversity / robustness candidate for the pool (10-ep pair, then the pool-level pmix
  read), not a recipe change. If the stale-val Δ is as negative as the val Δ, popularity is simply load-bearing and the
  -0.0012..-0.0023 private cost (st_pop) is the price of the family; nothing changes.
- **xdrop_ctxnd** load-bearing: the family is real worth (not only reliance) -> run **xdrop_ctxhn_hs** (10 columns) as
  the second-wave arm to complete the input ledger; ctx-type features remain a lever (the on-disk and 52 new families
  were null over G3, so "more ctx" means better ctx, i.e. the probes in findings §5-7 stay closed).
- **lnorm_batch** load-bearing (expected): K1 stays; the aux-normalisation mismatch (AUXF null) stays closed.
- **no_set** load-bearing: within-feed interactions matter -> cset-type priors at 10 ep deserve one more look (with the
  no_sim result in hand).

**DISPENSABLE** (|val Δ| <= 1 SE_cal, holdout >= -1 SE_cal, no slice < -2 SE_feed): candidate simplification; it changes
the recipe ONLY after a 10-epoch confirmation pair (E_BASE + HYG minus the component, seeds 951 / 952 vs
w5_e10_base_s951 / s952, same command as §3.1 with the flag removed; the unified rule applied to "arm not worse": val Δ
>= -1 SE, holdout >= -1 SE, no slice < -2 SE) AND an ensemble-level check (two members of the simplified recipe
replacing two base members in the pool: pool val / holdout not down):
- **no_set** dispensable -> adopt as the **speed lever** (~0.75x step time = more seeds per GPU-day for every future
  family; also fewer parameters, 0.71M vs 1.01M). This is the one ablation whose DISPENSABLE verdict has a direct
  action.
- **no_retr** dispensable (unexpected) -> a 0.85x lever and the end of retrieval scale-ups; confirm at 10 ep before
  believing it (the 10-ep cosine pair showed +0.0028 / +0.0030).
- **no_sim / no_domain / no_aux / lnorm_batch** dispensable -> simplification candidates; the 10-ep pair decides,
  and because K9 / domain act through the cooldown (under-dosed at 2 ep, §0.1 item 4) a 2-ep DISPENSABLE on no_aux
  or no_domain is NOT sufficient evidence for removal: the pair is mandatory and the CD harness read (re-cool of the F7
  roots without the component) is the cheaper first check (2-epoch cooldown from @77500, ~45 min per root).
- **xdrop_pop** dispensable (unexpected) -> the strongest possible private lever: the model does not need the only
  private-shifted family. Then: 10-ep pair + pmix at pool level; if confirmed, a pop-free family enters the pool first
  (diversity), and dropping popularity from the main recipe is a separate, later decision.
- **xdrop_ctxnd** dispensable -> the -0.011 replay reliance was redundancy (like the old cvraw-vs-cvnd pair), ctx_nd
  can go (simplification, -10 inputs, no speed gain); 10-ep pair before removal.

**INCONCLUSIVE** (everything else): the sign and the slices are reported; no recipe action. If |z_cal| is between 1 and 2
and the question matters for the next wave (no_retr, no_aux, xdrop_pop), add seed 25 to the screen (one base run + the
arm run, ~1.5 h on one GPU) before deciding whether a 10-ep pair is worth 6 GPU-hours.

### 3.3 Second-wave arms, conditional on tonight's readout (all off-by-default flags in nn_x/abl, same test pattern)

| if | then (2-ep screen unless stated) | code |
|---|---|---|
| retr32 or retr32u promotes | retr64; retr32u vs retr32 dedup read; retrieval key dim dk 128 | retr64: flag exists; dk: small |
| no_retr load-bearing (worth >= +0.002) and retr32 flat | retr_cat (second HistRetrieval over the candidate's same-category history items, zero-init Z) | new, ~40 lines + tests |
| xdrop_ctxnd load-bearing | xdrop_ctxhn_hs (ctx_hn 6 + ctx_hs 4) | `--xt_exclude`, no new code |
| xrel_more promotes | xt_rel_extra on ctx_hs_* / ctx_hn_* | no new code |
| no_domain load-bearing | dom_soft (soft p_random for M<50, INSIGHTS L6); DOMAIN-DIAG run | small; exists |
| no_aux load-bearing | aux_anneal 0 and aux_pos 1.0 listwise in the CD harness (stage B, roots 101-104) | flags exist |
| always (independent of the screen) | f1..f5 zero / permute replay on an existing checkpoint (GPU minutes, no training): the only never-read input family | hook in dense_features (replay only) |
| always | m01 / m04 in the CD harness (stage B, roots 101-104 vs w5_cd_m02ref_r<r>, CDSEED 900 + root) | flag exists |

### 3.4 Operational rules for tomorrow's follow-ups
- GPUs N only for this work; before EVERY launch: nvidia-smi shows < 4 compute processes and >= 60 GB free on that
  GPU; `CUDA_VISIBLE_DEVICES` = one id; 12 CPU threads; at most 3 of our processes per GPU. GPUs N and 7 are shared
  with w25 CatBoost-type jobs of another workstream (27-69 GB each, PIDs 4192xx at 00:31): count them, never touch them.
- Start every long driver with `setsid nohup ... < /dev/null &` (the first GPU test driver was lost to a shell exit).
- Stop only PIDs listed in nn_x/abl/RUNNING.txt (train.py PIDs verified via /proc cmdline; never a wrapper's $!).
- New code only as off-by-default flags in nn_x/abl with the AST + e2e bit-identity tests re-run (tests/test_abl.py
  cpu + gpu); never edit nn/, nn_x/r6, nn_x/w5.
- Pre-declare any new readout rule with `date -u` before the first run it applies to; the screen's rules stay as frozen.

## 4. Files
- Ledger: <repo>/nn_x/abl/ledger/ledger.md
- Harness: <repo>/nn_x/abl/{train.py, abl_arms.py, abl_lane.sh, launch_abl.sh, wait_abl_readout.sh,
  gate_and_launch.sh, abl_eval.py, tests/test_abl.py, tests/e2e_abl.py, tests/ref_train_w5.py.txt, tests/w5_source_md5.txt}
- Proofs: nn_x/abl/logs/test_abl_cpu.{log,json} (68/68), nn_x/abl/logs/test_abl_gpu.{log,json} (the re-run), freeze
  record nn_x/abl/logs/FROZEN_abl.txt (written by the gate at launch)
- Run state: nn_x/abl/RUNNING.txt, nn_x/abl/logs/{gate_and_launch.log, launch_abl.out, lane_abl_gpu{5,7}_{1,2,3}.out,
  wait_abl_readout.log, queue/, locks/}
- Readout (tomorrow): nn_x/abl/logs/abl_eval.txt (+ .json), abl_eval_prestale.txt, stale_pred_abl.stdout
- This report: <repo>/reports/ABLATION_PLAN_1113.md
