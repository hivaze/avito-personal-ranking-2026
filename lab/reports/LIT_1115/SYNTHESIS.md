# LIT_1115 / SYNTHESIS: mixture-of-experts and graph networks. Checked? Worth trying?

Written 2026-11-15 15:30-16:00Z by the synthesiser of workflow a workflow run. CPU only, no GPU process, nothing trained, no submission file.
Inputs: `reports/LIT_1115/{int_moe,int_gnn,lit_moe,lit_gnn}.md` (read in full), `reports/CURRENT_STATUS.md` sections 1-4 and the log to 14:34Z, `reports/COVERAGE_MAP.md` (all 336 lines).
Own work directory: `nn_x/w12/lit/syn/` (rules written in `DECLARATION.txt` before each read). Nothing else was written or edited.

Marks: **[M]** measured by me today (log named); **[D]** read in a project file; **[W]** read on the web by me today (4 spot checks); **[W-lit]** web claim taken from the two literature reports, not re-read by me; **[I]** inferred. "+-" is one standard error of a per-feed paired difference unless stated.

Terms used below:

| term | meaning |
|---|---|
| pool, file | pool = member-equal mean of per-feed z-scores of trained rankers; file = 0.5 z(fine-tuned pool) + 0.5 z(train-only pool) |
| popularity-light students | students trained without the 8 popularity / co-visitation inputs; mixed into the file at weight 0.2 (the form of both current picks) |
| cold row | candidate with no interaction count: 37.9% / 42.8% / 45.6% of val / public / private rows [M, int_gnn] |
| witnesses | val; pmix = val reweighted to the public or private feed mix; stale-val = val re-predicted with private-like stale inputs; lagged holdout; in-period holdout (wrong in sign for popularity contrasts, never used alone here) |
| residual lens | a column is added to a finished pool with a weight fitted on other users. It under-reads a trained input: a block worth +0.0085 as a trained input read +0.0009 to +0.0022 [D, COVERAGE_MAP l.323] |
| frontier law | gain of an equal mix = 0.0186 x (correlation margin) - 0.378 x (strength gap) [D, `reports/SOLVER_POWER_1113.md` l.22] |
| P | my probability that the TRUE effect of an 8-member arm against its same-seed reference pool is >= +0.0005 on val [I] |

## 0. Verdict

1. **MoE: checked partly. Graphs: checked partly.** In both cases the half that was skipped is the expensive network, and in both cases the cheap half answers it.
2. **Try none.** No candidate reaches a prior of 0.15 for +0.0005; the highest is 0.03 (section 3). The bar is missed by a factor of 5, not narrowly.
3. GPU time is not the reason. the GPUs x 24 days is about 4,600 GPU-hours and an 8-seed arm costs 12-31. The reason is that every candidate either lacks a mechanism for the four measured bottlenecks or has that mechanism already measured at zero.
4. New today [M]: the one graph construction the two graph reports left open (collaborative scores for cold rows read through content neighbours) shows no detectable information beyond the current file on cold rows: likelihood-ratio p 0.34 (1 df) and 0.47 (15 df); re-ordering cold rows with it reads -0.00030 +- 0.00020. The evidence that kept it alive was a rank-residual AUC (0.545 in int_gnn's form); in my form that statistic reads 0.50-0.53 on a placebo that has no information.
5. New today [M]: the fresh-density columns (lit_gnn S2) add nothing to the current file in the residual lens (-0.00030 +- 0.00031, likelihood-ratio p 0.20); the only learned neighbour-aggregation layer ever trained here (cset) has no support on the private mix (-0.00016 +- 0.00105).
6. The MoE-type idea that did pay is already in both picks: two "experts" with different INPUT VIEWS at a constant weight (popularity-light admixture, public +0.00052 and +0.00041 [D]). Its extension (more views against a matched control) is on the GPUs now, reads due 16:30-17:15Z [D].
7. If the owner wants a measurement behind each "no" instead of an argument, section 4.2 gives two closure runs. They are optional filler with an expected gain of zero, not recommendations.

## 1. The two answers

**Mixture of experts. Checked: PARTLY.**
- Gates and routers between finished models were tested in 8 studies plus candidate-level gates today, all null; experts inside the network were trained once, in a test that was not fair (2 of 10 epochs, old optimiser, 2 seeds).
- Three numbers: in-network MoE val **+0.00027 +- 0.00047**, and -0.00034 +- 0.00052 after the fine-tune [D]; learned routers **AUC 0.494-0.511 against a permuted null of 0.496-0.503** [D]; in-network diversity margin **0.0032 against 0.027 needed** for +0.0005 by the frontier law, i.e. worth +0.00006 [D, I].

**Graph networks. Checked: PARTLY (graph information yes, a graph neural network no).**
- 5 graph-type signals are in the model and 16 were built and closed; a GNN was never trained, it was closed by argument on 11-13 [D, int_gnn].
- Three numbers: a GNN on the interaction graph has **no node for 42.8% / 45.6%** of public / private candidate rows, and **two hops already reach 97%** of the rest [M, int_gnn]; the one inductive form left open reads **-0.00030 +- 0.00020** on cold rows with likelihood-ratio p 0.34 [M, this report]. The cleanest published ablation agrees: removing the graph embedding next to a sequence encoder changes three scores by -0.0001 / -0.0001 / +0.0002 [W, arXiv 2508.06970 table 6, checked today].

How each half meets the four measured bottlenecks:

| bottleneck [D] | what MoE offers | what graphs offer |
|---|---|---|
| stale popularity inputs: pool loses 0.0033-0.0042 under private-like staleness | only a staleness-aware gate (M4); its parent recipe failed at file level on 3 member sets today | a GNN adds more ID-keyed inputs of the same age; only the fresh-density columns (G2) are lag-0, and they read null |
| 42.8% / 45.6% cold rows | cold gates and cold experts: gate AUC 0.65-0.67 but value-weighted AUC 0.52-0.53; row-level gate -0.00011 +- 0.00003 | no node for a cold row; the content-completed form (G1) shows no detectable information beyond the file |
| trees input-limited (0.3815-0.3865 against 0.4016+) | not applicable | a GNN-only scorer reaches 0.33-0.34: further from the pool than the trees that already fail the member rule |
| ensemble 0.0002 below its infinite-member level | in-model diversity margin 0.0032, 8 times too small | a member must be within about 0.002 of pool strength; graph scorers are 0.04-0.07 below |

## 2. What I measured today

All reads are on val, CPU only. Rules and their time stamps: `nn_x/w12/lit/syn/DECLARATION.txt` (block 1 before any number; blocks 2-6 are information-only amendments, each written before its read).

### 2.1 Neighbour-completed collaborative scores on cold rows (lit_gnn S1 = int_gnn form C)

Data: int_gnn's stored neighbour file for 3,000 random val feeds (107,992 cold rows, 1,358 cold positives; 1,076 feeds with a cold positive and a cold negative). The sample had already been read by int_gnn, so this is exploratory. Columns: x1 = cosine-weighted mean of the 3-step walk score of the 5 nearest warm train items (cosine >= 0.90), x2 = the same for the 5-step walk. Pool F = the scored file FD16P20 (val 0.40271); pool W = W16 (0.40158).

| read | x1, pool F | x2, pool F | x1 / x2, pool W | log |
|---|---|---|---|---|
| AUC among cold rows: the pool alone / the column alone | 0.749 / 0.609 | 0.749 / 0.613 | 0.751 / same | `gate_A.log` |
| A1: AUC of the column after a linear rank regression on the pool | 0.528 +- 0.010 | 0.541 +- 0.010 | 0.528 / 0.541 | `gate_A.log` |
| A6: the same statistic on a placebo with no information given the pool (column permuted among rows adjacent in pool order, blocks of 4, 10 draws) | 0.504 (max 0.520) | 0.525 (max 0.538) | 0.509 / 0.531 | `placebo_A.log` |
| A5: likelihood-ratio test, conditional logit over cold rows, pool vs pool + column (1 df) | LR 0.92, p 0.34 | LR 0.66, p 0.42 | p 0.36 / 0.43 | `lr_A.log` |
| A5: pool vs pool + all seven columns in z and rank form (15 df) | LR 14.7, p 0.47 | - | p 0.51 | `lr_A.log` |
| A2: cold rows re-ordered by pool + w x column, warm rows fixed; w nested by user | **-0.00030 +- 0.00020** | -0.00036 +- 0.00019 | +0.00019 +- 0.00028 / -0.00003 +- 0.00023 | `gate_A.log` |
| A3: the same with the column permuted inside the feed's cold rows, 20 draws | -0.00048 .. +0.00046 | -0.00040 .. +0.00045 | -0.00063 .. +0.00065 | `gate_A.log` |
| A4: best linear use of all seven columns inside the cold rows, fitted IN SAMPLE (optimistic bound) | +0.00017 +- 0.00037 | - | +0.00012 +- 0.00042 | `ceiling_A.log` |
| A4: the same, nested by user | -0.00047 +- 0.00040 | - | +0.00011 +- 0.00043 | `ceiling_A.log` |

Reading:
- By the words declared in block 1 the result is INCONCLUSIVE (A1 is above 0.515, so not "kill"; A2 is below +0.0003 and A1 below 0.530, so no confirmatory stage). Stage B (the other 11,423 val feeds, about 50 CPU-minutes) was therefore not run.
- The NDCG reads, declared (A2) and information-only (A4), point the same way. A2 is "slot-preserving": warm rows never move, so a feed without a cold positive changes by exactly zero and the cold/warm trade that sinks every collaborative residual is removed by construction. Even so the gain is not above its placebo, and the in-sample curve of x1 never exceeds +0.00004 (pool F) / +0.00019 (pool W) and is negative from w = 0.1 on.
- A1 is not a clean statistic. A placebo column that keeps its relation to the pool and (almost) nothing beyond it reads 0.504-0.531; my explanation is that rank(column) is not linear in rank(pool) [I]. Real minus placebo is +0.010 to +0.024, which is 1.0-2.4 standard errors of the real read. int_gnn's "partial AUC 0.545 +- 0.010" uses the same statistic with two content inputs in place of the pool, so it overstates the case in the same way [I].
- Limits: one sample of 3,000 feeds, val only, residual lens; a paired SE of 0.0002-0.0004 cannot exclude +0.0005 as a trained input. This is not lit_gnn's pre-declared stage 0 (that one uses the windowed count, same-cluster neighbours and the lagged holdout).

### 2.2 Fresh-density columns on the current file (lit_gnn S2, family i)

The record had these columns only against the old pool G3 (-0.00035 [D]). Read on all 14,423 val feeds: conditional logit of `nn_x/w12/fresh/pregate` (control = pool, pool x log M, cold flag and its two interactions; arm = control + the 10 columns `trd_{kde,n85,n95,skde,sn95}_{sday,s6h}` in raw and rank form), 5 folds by user x 3 fold seeds.

| read | pool F | pool W | log |
|---|---|---|---|
| in-sample likelihood ratio, arm vs control (20 df) | 25.1, p 0.20 | 24.3, p 0.23 | `trd_read.log` |
| nested arm minus control | **-0.00030 +- 0.00031** | -0.00045 +- 0.00030 | `trd_read.log` |
| pmix public / private | -0.00058 +- 0.00038 / -0.00057 +- 0.00047 | -0.00039 / -0.00029 | `trd_read.log` |
| placebo (column vector permuted inside feed and cold status, 10 draws, fold seed 0) against the real read on that seed | -0.00066 .. -0.00007 against -0.00024 (above 8 of 10) | - | `trd_read.log` |

Declared word: NO SIGNAL IN THE RESIDUAL LENS. The real columns sit about 1.3 placebo standard deviations above the placebo mean, which is the size of a hint, not of evidence.

### 2.3 cset under a staleness-type witness (flag of int_gnn, row 6)

cset is the only learned neighbour-aggregation layer over an explicit kNN graph ever trained in the ranker. It was closed on val and the in-period holdout and never re-read. With `src/pmix.py`, 6-member train-only pool (seeds 101-106) against the same-seed reference pool [M, `cset_pmix.log`]: val -0.00010 +- 0.00066, pmix public -0.00007 +- 0.00084, **pmix private -0.00016 +- 0.00105**. The closure stands. No stale-val or lagged-holdout read was made (they need GPU inference).

### 2.4 Facts verified for the cost column [M]

- The sparse-MoE block and its flags exist only in `nn_x/attn/{model,train}.py` (12 and 7 matching lines); `nn/`, `nn_x/w5/`, `nn_x/abl/` and the two full-data trainers have none.
- All 20 `trd_*` files exist for train, val and test; no trainer or data module of `nn`, `nn_x/w5`, `nn_x/abl` references them.
- `nn_x/abl/train.py` has `--xt_add` (append input columns) and asserts that it is not combined with `--init` (l.299-300). An added-input arm is therefore a from-scratch 10-epoch arm, never a 1-GPU-hour re-cool.
- `data/feats_new/train/hz_lag.npy` (the lag of a train row under the stale view) is on disk.

## 3. Every candidate, priced

Cost basis [D, task brief]: one 10-epoch member 1.2-3.5 GPU-h, one fine-tune 0.33 GPU-h. An 8-seed arm with fine-tunes = **12-31 GPU-h**; a 2-seed first stage = 3-8 GPU-h. The reading rule in force needs 8 against 8: it passes 0.25% of nulls and 56% of true +0.0005 effects [D, COVERAGE_MAP l.324]. One paired val read has SE 0.0003-0.0005, so a true zero shows +0.0005 or more in 5-16% of single reads [I]; no kill rule below accepts a val number alone.

Two kill-rule templates, to be copied into `nn_x/w5/logs/E_kd_predeclared.txt` with `date -u` before the first run:

- **K-ARCH** (architecture arms; not a popularity contrast). Reference = the existing wide students of the same seeds (951-958), same teacher, same fine-tune.
  - Stage 1, seeds 951-952: STOP if the mean member delta is below 0 on val AND on pmix-private, or on an implementation fault (flag-off run not bit-identical; for MoE any block whose largest expert load exceeds 20 times its smallest at step 60,000).
  - Stage 2, seeds 953-958 added, read at 8-pool level in the final form: PASS iff val >= 0 AND pmix-public >= 0 AND pmix-private >= 0 AND stale-val >= -1 SE AND pooled z over val and the in-period holdout >= 2 at matched N. Pooled z 1.6-2 with all signs right: 8 more seeds. Anything else: CLOSED for good.
  - The in-period holdout enters only inside the pooled z. The arm is trained as held-out members (not full-data twins), so that the holdout and stale-val can be read.
- **K-STALE** (arms that change how cold or stale rows are scored). Witnesses: val, stale-val, pmix-public, pmix-private, lagged holdout; the in-period holdout is printed and has no clause.
  - Pool level, 8 against 8: PASS iff val >= 0 AND pmix-private >= 0 AND stale-val >= 0 AND z >= 2 on val or on stale-val.
  - File level (the lesson of hz: an arm-level pass that vanished in the file): the file with the arm's members in place of their reference members, admixture 0.2 kept, minus the reference file: val >= 0 AND pmix-private >= 0. A pool-level PASS with a file-level FAIL is CLOSED.

### 3.1 Mixture-of-experts family

| # | idea (source) | mechanism | link to a measured bottleneck | the record against it | cost | P and why | kill rule |
|---|---|---|---|---|---|---|---|
| M1 | Sparse MoE feed-forward under the current recipe (int_moe section 4; lit_moe N1) | The feed-forward of the 4 candidate blocks becomes 1 shared + 8 routed experts (top-2), so each candidate token uses its own sub-network at equal compute | None. It buys parameters; no route to staleness or cold rows | The one run: +0.00027 +- 0.00047, -0.00034 after fine-tune. Under distillation 0 of the 7 arms of the capacity row pass (dense feed-forward x4: -0.00068 val / -0.00088 holdout; width 192 -> 224 null). The router did learn category routing (0.33-0.50 nats) and it was worth 0. Margin 0.0032 [D] | Port 0.5-1 day (block, 2 flags, load-balance update, flag-off bit-identity test); 12-31 GPU-h; first stage 3-8 | **0.02.** Strength: 0 of 9 architecture changes pass under distillation, prior mean about -0.0002. Diversity: +0.00006 by the frontier law | K-ARCH |
| M2 | In-network context gate, POSO / PPNet / LHUC form (lit_moe N2; int_moe section 4) | A zero-initialised sigmoid gate fed only with feed context (anonymous flag, log history length, log M, feed cold share) rescales hidden units | Cold-heavy and short feeds, in principle | The same covariates as a router between pools equal the permuted null. The adopted attention gate is its near-equivalent (+0.00065, mostly absorbed by the fine-tune). The published mechanism (a rare group drowned by ID embeddings) is absent: no ID table here. PPNet is below shared-bottom on 4 of 6 public sets [W-lit] | About 30 lines, 0.5 day; 12-31 GPU-h | **0.02.** Its own CPU pre-gate is on record and failed | CPU pre-gate (router AUC above null + 0.02 on val): FAILED 11-13 and 11-15 [D]. If run anyway: K-ARCH |
| M3 | De-correlation loss between 2-3 parallel heads, D-MoE (lit_moe N3) | Several heads in one model are penalised for agreeing and then averaged: diversity inside one member | Ensemble level, but the pool is 0.0002 below its infinite-member level | Weight-sharing heads were 0.999-correlated and null (+0.00009 +- 0.00042). Students of one teacher share one mean function (0.999+). The paper has no seeds and no comparison with averaging separate models [W-lit] | 1 day; 12-31 GPU-h | **0.01.** The most it can reach is the 0.0002 of headroom | K-ARCH plus an in-pool replay (8 of 16 members swapped) |
| M4 | Staleness-aware gate: lag input on stale training views, or per-feed blanking of the popularity inputs with an indicator (int_moe N7; `reports/STALE_1115/veto.md` item 8) | Train on stale views and tell the model how stale its counters are, so one network holds a fresh-counter and a stale-counter regime | The largest measured loss (0.0033-0.0042). The only MoE-type idea with a mechanism for it | Its parent without the lag flag (hz) passes stale-val as a 16-run arm (+0.00120 +- 0.00045) and fails in the file on top of the popularity-light admixture on three member sets (-0.00047 +- 0.00035 on the last): the two substitute. Popularity-input dropout: val -0.00066 +- 0.00048, stale-val +0.00001. Cold rows already carry an implicit flag (all 8 inputs zero on 100% of them). The lag is one number per feed and almost one per split, so its use at test time is an extrapolation from simulated lags [D, I] | Lag column on disk; needs the stale-view pipeline and a from-scratch arm (`--xt_add` cannot be used with `--init`); 12-31 GPU-h, 1 day | **0.02** on val (val is 5.7 days stale; the target is private). P(file gain >= +0.0003 on the private-like witnesses) 0.05 | K-STALE with the file-level clause. Re-open trigger: a finished hz2n / hzt run (on disk, unread [D]) that reads >= 0 on val and >= +1 SE on pmix-private at FILE level |
| M5 | Experts by feed size, feed-length cooldown specialists (int_moe section 4, N3) | Separate sub-networks or cooldowns for short and long feeds | Short feeds (M < 50) | The feed-size rule read -0.00015 on public today. No usable random-order training feeds with M < 50 (exploration pool null, sub-feed augmentation mismatched). Every cooldown data-selection arm hurt [D] | 12-31 GPU-h | **0.01** | K-ARCH; not worth stage 1 |
| M6 | Cold / warm candidate experts inside the network (int_moe section 4) | A sub-network for cold candidates | Cold rows | Feed-level cold gate -0.00005 +- 0.00004; row-level gate -0.00011 +- 0.00003; popularity-light weight on cold rows only -0.0006 (z -2.1 / -2.3). The admixture pays through the order among WARM rows [M, int_moe] | 12-31 GPU-h | **0.01** | K-STALE |
| M7 | Per-vertical experts or adapters, trained (int_moe N2) | One expert per item vertical | None | 99.2% of feeds mix verticals; a per-vertical correction of the pool reads -0.00095 +- 0.00039; the M1 router found verticals for 0 [D] | 12-31 GPU-h | **0.01** | CPU pre-gate FAILED 11-13 |
| M8 | Trajectory-cluster experts or FiLM, trained (int_moe N6) | One expert per cluster of user histories | None | 0 of 8 per-cluster re-weighting arms; the best (+0.00022 +- 0.00025) is below its random-label control [D] | 12-31 GPU-h | **0.01** | CPU proxy FAILED 11-13 |
| M9 | Multi-task experts with event-type heads, MMoE / Progressive Layered Extraction (int_moe N4) | Task gates over shared experts to separate conflicting objectives | None | No per-type label at train time. Main and auxiliary gradients are near-orthogonal (cosine +0.007..+0.014): no conflict to separate. MMoE and PLE are below shared-bottom on 4 of 6 public sets [D, W-lit] | - | **< 0.01** | CPU pre-gate FAILED 11-13 |
| M10 | MoE over the history encoder (int_moe section 4) | Experts per event-type group in the history layers | None | GRU about equals the transformer encoder; a second history layer is <= 0 under distillation; surface-restricted history views -0.003..-0.005 [D] | 12-31 GPU-h | **0.01** | K-ARCH |
| M11 | Full weight-sharing ensemble, adapters in every layer (int_moe section 4) | k members share one trunk | Ensemble level | Bounded above by the 0.0002 the pool lacks to its infinite-member level [D] | 12-31 GPU-h at about 1.5x step time | **0.01** | K-ARCH |
| M12 | Scenario or traffic-domain towers (int_moe N5) | One tower per domain | None | The domain is constant on val and test [D] | - | **0** | none needed |
| M13 | Sparse MoE grown in the cooldown only (int_moe N1) | Experts diverge during the last 20% of training | None | 32 stored cooldown arms: margin median 0.0016, maximum 0.0104 [D] | 1 GPU-h per re-cool | **< 0.01** | closed on stored arms |
| M14 | More gates or routers between pools | A learned or rule-based weight per feed or per row | - | 8 studies, 131,547 declared rules, a learned gate, row-level gates: all at or below a constant weight; fitted-gate optimism 0.0004-0.0010; public -0.00015 for the simplest rule [D] | CPU | **0** | closed |

### 3.2 Graph family

| # | idea (source) | mechanism | link to a measured bottleneck | the record against it | cost | P and why | kill rule |
|---|---|---|---|---|---|---|---|
| G1 | Neighbour-completed co-visitation for cold rows as trained inputs (lit_gnn S1; int_gnn C) | Read a candidate's co-visitation or walk score through its nearest warm content neighbours, so a cold row gets a collaborative input whose zero share does not move with the lag | Cold rows and staleness: 47-48% of cold rows get a link and the share does not fall from val to private [M, int_gnn] | Section 2.1: no detectable information beyond the file on cold rows (p 0.34 / 0.47; -0.00030 +- 0.00020; best linear use in sample +0.00017 +- 0.00037). Cluster-level form -0.0004 +- 0.0011; neighbour click statistics +0.00004 nats; cset null at 10 epochs and on the private mix. The item vectors are already a two-tower output [M, D] | Train-row build 5 CPU-h (same-cluster search) to 45 (exact) plus 1-1.5 days for causal sources and parity checks; 12-31 GPU-h | **0.03** (0.05 before section 2.1). Only route left is a non-linear use that the residual lens cannot see | CPU pre-gate run today: INCONCLUSIVE by its letter, no positive NDCG read. If run anyway: K-STALE |
| G2 | Fresh neighbour densities as trained inputs of the popularity-light students (lit_gnn S2) | Same-day demand and supply density around the candidate in embedding space, counted from the val / test-period feeds themselves | Staleness: the only collaborative signal with lag 0 at test time | Section 2.2: p 0.20, -0.00030 +- 0.00031 on the current file; -0.00035 on G3. Near-duplicate coverage 8.6-10.4% of cold rows, exact item 0.9% of rows. Four extra inputs once cost -0.0019 at 10 epochs. Only the popularity-light students can host it (in the base students the fresh train-row counters dominate), so a pool gain reaches the file at 0.2-0.3 [M, D, I] | No build (files on disk), no code (`--xt_add`); 12-31 GPU-h | **0.03** at the popularity-light pool; P(file >= +0.0003) 0.01-0.02 | K-STALE |
| G3 | Multi-hop walk score or ALS factors as train-stage inputs (int_gnn B, rows 7, 9, 26) | A 5-step item-user-item walk from the history: the two-hop signal of a 2-layer GNN without training one | None in the right direction: ID-keyed, zero on cold rows, reach falls with the lag (two-hop 57.0% val -> 49.4% private) | In the territory it adds over stored co-visitation its AUC is 0.58-0.59 against 0.64 for a content input already in the model; best of 42 residual arms +0.00046 +- 0.00032; every collaborative residual trades cold positives for warm ones; four more collaborative inputs -0.0019 at 10 epochs [M int_gnn, D] | 2 CPU-h causal build, 0.5 day; 12-31 GPU-h | **0.03** on val, with a negative expected sign on the private-like witnesses | K-STALE |
| G4 | Trained transductive GNN (LightGCN, PinSage, TwHIN), embeddings or scores into the ranker (int_gnn A; lit_gnn R1, R4) | Learned message passing on the user-item graph | None: no node for 42.8% / 45.6% of test rows, 70% of test users have no node, edges 9-11 days old | G3 is its signal and reads null. ID table in the ranker -0.0022 / -0.0051; ID part of the encoder score +0.0008 val / 0.0000 holdout. Tuned linear and walk baselines match published GNNs [W-lit]; removing the graph embedding next to a sequence encoder: -0.0001 / -0.0001 / +0.0002 [W] | Per-day causal graph cuts for 3.15M train feeds, a 95M-node model, then an 8-seed arm: 60-100 GPU-h, 3-4 days [I] | **0.01** | Its CPU pre-gate is G3's residual read: done by int_gnn, no arm of 42 at 2 SE |
| G5 | Session-graph history encoder, SR-GNN type (int_gnn D; lit_gnn R2) | A graph over the items of one history replaces attention | None | Tuned kNN and sequential rules beat all eight session GNNs on MRR [W-lit]. Here GRU equals the transformer and DIN loses; histories have no timestamps [D] | 1-2 days; 12-31 GPU-h | **0.01** | K-ARCH |
| G6 | GNN-only ranker as a pool member (int_gnn E; lit_gnn R13) | A second model family for diversity | Ensemble level | Graph scorers reach 0.33-0.34 alone against members at 0.40; trees at 0.3865 already fail the member rule [D, M int_gnn] | 60+ GPU-h | **0** | strength gate |
| G7 | Graph centrality: PageRank, Katz (int_gnn F) | Global importance of an item | None | Two-hop counts equal degree, and degree is an input [M int_gnn] | CPU | **0** | none needed |
| G8 | cset re-read on a staleness witness (int_gnn flag) | - | Cold rows | Section 2.3: pmix-private -0.00016 +- 0.00105 [M] | done | - | closed |

## 4. Recommendation

### 4.1 Try none

Nothing clears a prior of about 0.15 for +0.0005. The ranking by prior is G1 = G2 = G3 = 0.03, then M1 = M2 = M4 = 0.02, and for each of the three graph items the 0.03 comes with a reason to expect less at file level (G2: leverage 0.2-0.3; G3: wrong sign under staleness; G1: no detectable information in section 2.1).

The cheap pre-gates the brief asks for exist already for every item that could have one, and all are negative or empty:

| item | CPU pre-gate | result |
|---|---|---|
| M1 | diversity margin of the archived MoE runs; dense capacity under distillation | 0.0032 against 0.027; 0 of 7 [D] |
| M2 | router on the gate's own covariates | AUC equal to the permuted null [D] |
| M4 | arm-level stale-val of the parent recipe, then its file-level read | passes, then fails in the file on three member sets [D] |
| M5-M9 | feed-size rule on public; row-level and feed-level gates; per-vertical residual; cluster weights; gradient cosine | all null or negative [D, M int_moe] |
| G1 | section 2.1 | no detectable information beyond the file [M] |
| G2 | section 2.2 | no signal in the residual lens [M] |
| G3, G4 | walk-score residual, 42 arms | none at 2 SE [M int_gnn] |

What would change this answer:
- A label-free covariate whose router AUC is clearly above the permuted null on val (re-opens M2 and M14).
- A file-level read of a stale-view recipe that is >= 0 on val and >= +1 SE on pmix-private (re-opens M4).
- A null on today's view-diversity read would NOT change it; a clear positive there would argue for more input views, which is the member of the MoE family that pays, not for a gated network.

### 4.2 Optional closure runs (only if the owner wants a measurement instead of an argument)

Expected gain zero. Run only on GPUs that would otherwise idle, after the running reads. At most these two; I do not recommend a third.

1. **C-MoE: sparse-MoE wide students (M1).** It is the single run that turns "in-network MoE: trained once, unfairly" into a measurement.
   - Smallest decisive form: port by write-new + mv, flag-off bit-identity test, then seeds 951-952 only (3-8 GPU-h) under K-ARCH stage 1. Continue to 8 seeds only if stage 1 does not stop.
   - Honest forecast: stage 1 stops or reads inside noise in about 9 cases of 10 [I].
2. **C-Graph: one input arm on the wide popularity-light students (G2, optionally G1 in the same arm).** `--xt_add` of 4-6 fresh-density columns that are on disk (the two `kde` and two `skde` columns plus `trd_n95_sday`, `trd_sn95_sday`); 8 seeds against the existing wide popularity-light students of the same seeds; K-STALE including the file-level clause.
   - There is no smaller decisive form: an added input cannot be a re-cool, and the CPU lens has been used (section 2.2).
   - A null closes routes 3 and 5 of lit_gnn section 4 and with them the graph class. Add the G1 columns only if the lead accepts the 5 CPU-h build for the sake of that closure.

A trained GNN (G4) is not on this list at any price: its two-hop signal and its reach were measured today without training one.

## 5. What I would not do, and why

| popular method | why not here |
|---|---|
| LightGCN, NGCF, SGL, SimGCL, graph transformers on the interaction graph | Transductive: no node for 42.8% / 45.6% of test rows; the rest carry edges 9-11 days old, the same defect that costs 0.0033-0.0042 today. Tuned EASE / RP3beta / kNN match them in three reproducibility studies [W-lit]; ALS, item2vec, group-level EASE and RP3beta read 0 to -0.0005 here [D] |
| PinSage / TwHIN / OmniSage embedding pre-training as extra item inputs | Their gains are against systems without a learned item vector and with a daily graph refresh [W-lit]. The item vectors here already are a production two-tower output; next to a sequence encoder the graph embedding is worth nothing [W] |
| Session-graph encoders (SR-GNN, GCE-GNN), temporal graph networks | Beaten by tuned neighbour rules [W-lit]; node memory is transductive; no dense event stream exists after Sep 19 [D] |
| A GNN as a diversity member (as in OTTO 7th place) | Needs a member near pool strength on warm IDs in a same-period test set. Neither holds |
| MMoE, PLE, HoME, multi-task expert stacks | They repair conflicts among 8-20+ tasks. One task here and an auxiliary whose gradient is orthogonal to it [D] |
| STAR, PEPNet, AdaSparse, M3oE (multi-scenario gating) | Need a scenario that varies at inference; ours is constant. In the independent 6-dataset benchmark no gating model wins everywhere and the best gain on sets above 10M interactions is +0.003 AUC [W-lit; benchmark scope checked today] |
| RankMixer / TokenMixer-Large per-token feed-forward and sparse MoE, SORT, OneTrans | A way to add parameters at constant serving cost at 1e10-1e12 rows and 0.5-15 billion parameters [W: TokenMixer-Large abstract, "efficient parameter expansion", 7 and 15 billion]. There is no latency budget here and dense scale-ups are negative under distillation [D] |
| POSO, Cold & Warm Net and other cold-start gates | User cold start with ID tables. Here the cold-share covariate predicts a cold click and anti-predicts the payoff of boosting cold items in every quintile [D] |
| Learned routers or per-feed weights between pools, feed-size rules | 8 studies; fitted-gate optimism (0.0004-0.0010) is larger than the gains sought; public -0.00015 for the simplest rule [D] |
| A second look at any oracle ("perfect per-feed switch is worth +0.019") | About half of every such oracle is noise selection (same-recipe reference +0.0092) and the rest needs the label [M, int_moe] |
| Graph centrality, more co-visitation variants, cross-feed co-exposure | Degree is an input; 6 co-visitation variants are closed; co-exposure covers 0.1-4.5% of rows [D] |

## 6. Where the four input reports are out of date or need a correction

1. int_moe names the feed-size rule as "the only survivor" of the gate class with its public pair unread. It was read at 14:34Z: -0.00015 (-0.5 SE), closed [D, `reports/CURRENT_STATUS.md` section 1]. The gate class now has no survivor.
2. lit_gnn lists S1's key quantity as unmeasured and gives P(stage 0 PASS) 0.25. int_gnn measured a version of it, and section 2.1 conditions it on the pool: no detectable information. lit_gnn's own stage 0 (windowed count, same-cluster neighbours, lagged holdout) remains unrun.
3. int_gnn's "partial AUC 0.545 +- 0.010, 4.5 SE above 0.5" should not be quoted as evidence of information beyond the model: the statistic has a positive bias (section 2.1, read A6), and against the pool a likelihood-ratio test gives p 0.34-0.47.
4. The brief's "43-49% cold" is 42.8% / 45.6% over all public / private rows by the stored count [M, int_gnn]. Another definition in use today (no train impression) gives 34.6% val / 46.2% test [D, `nn_x/w12/fresh/pregate/DECLARATION.txt`].
5. COVERAGE_MAP l.61: "PLE" is piecewise-linear numeric encoding and "TabM" a weight-sharing ensemble; the only MoE in that row is the 2-epoch run. COVERAGE_MAP l.280: the holdout figure "-0.0025, z -6.0" equals route share x the known holdout bias; that closure rests on its val-nested null [D, int_moe].
6. A sibling pre-gate running in parallel (fresh exposure counts from the val + test feeds, `nn_x/w12/fresh/pregate`, not mine) reads its decision arm at -0.00044 +- 0.00015 in the same lens [D, its PROGRESS.md at 15:13Z, interim]. It agrees in sign with section 2.2.

Rows the lead may add to COVERAGE_MAP (this report edits nothing):

| row | text |
|---|---|
| new: neighbour-completed collaborative scores on cold rows, residual lens with warm rows fixed | NO SIGNAL 11-15 (CPU, 3,000 val feeds): -0.00030 +- 0.00020, LR p 0.34 / 0.47, in-sample ceiling +0.00017 +- 0.00037; trained-input form not run, P 0.03. `reports/LIT_1115/SYNTHESIS.md` 2.1 |
| new: `trd_` fresh-density columns on the current file | NO SIGNAL 11-15 (CPU, all val feeds): -0.00030 +- 0.00031, LR p 0.20. SYNTHESIS 2.2 |
| l.107 / l.111 note (cset) | 6-pool pmix-private -0.00016 +- 0.00105: closure also holds on the private mix. SYNTHESIS 2.3 |
| l.61 and l.178 notes | in-network MoE and a trained GNN have no run under the current recipe; priced at P 0.02 and 0.01, not recommended. SYNTHESIS 3 |

## 7. Not checked, and limits

- No model was trained and no GPU was used. Every P is my estimate [I]; none is a measurement.
- My reads are val-only. No stale-val, lagged-holdout or in-period-holdout view exists for the columns of sections 2.1 and 2.2.
- Section 2.1 uses one sample of 3,000 val feeds that int_gnn had already read, with int_gnn's columns (whole-sequence walk, catalogue-wide neighbour bank). The confirmatory stage on the other 11,423 feeds was not run, by the declared rule.
- Sections 2.1 and 2.2 are residual-lens reads. They bound a linear use next to a finished pool, not a trained input (COVERAGE_MAP l.323).
- Literature: I re-read four sources only (arXiv 2508.06970 table 6; 2404.08660, 2412.17374 and 2602.06563 abstracts). Every other external number is taken from lit_moe / lit_gnn, whose authors state that about half their sources were read as abstracts or summaries.
- I did not re-run any study of the two audits (routing, cold gates, 78 combiners, graph reach, walks); their numbers are quoted as they report them.
- GPU-hour costs use the brief's 1.2-3.5 per member; the 60-100 GPU-h for a GNN and all "days of work" are my estimates.
- The result of the view-diversity read due 16:30-17:15Z was not available when this was written.
- Rule note: `trd_read.py` averaged 14.3 busy cores for 105 seconds, above the 12-thread cap (the numerical library's thread pool ran next to a second one). Noticed after the run; peak memory 1.8 GB; recorded in `DECLARATION.txt` block 7.

## 8. Files

All in `nn_x/w12/lit/syn/`:
- `DECLARATION.txt`: rules and results, 7 time-stamped blocks.
- `gate_read.py`, `gate_A.log`, `gate_A.json`, `delta_A_{F,W}_x1.npy`: reads A1-A3.
- `ceiling_read.py`, `ceiling_A.log`, `ceiling_A.json`: read A4.
- `lr_read.py`, `lr_A.log`: read A5. `placebo_a1.py`, `placebo_A.log`: read A6.
- `cset_pmix.log`: read A7.
- `trd_read.py`, `trd_read.log`, `trd_read.json`, `trd_read.time`, `trd_delta_F.npy`: read A8.

To extend section 2.1 to all val feeds (about 50 CPU-minutes, 14 GB, one process): copy `nn_x/w12/lit/int_gnn/{g_common,s2_graph,s3_walk,s4_content,s5_coldsignal}.py` into a new directory, run `s5_coldsignal.py 14423` there, then `gate_read.py <that directory>/s5_cold_val.npz B`.
