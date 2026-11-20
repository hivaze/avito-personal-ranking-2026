# COMPONENT LEDGER of the current ranker recipe (nn_x/abl/ledger)

Written 2026-11-12T23:52Z by the ledger step (CPU only; no GPU job launched, nothing under nn/, nn_x/r6, nn_x/w5 edited).
Purpose: for every component / setting of the live recipe, record what its contribution was measured to be, on which
recipe and horizon, and where the measurement is stale or missing; then compile the scale-up candidates and rank the
<= 12 arms for a 2-epoch paired screen on the wave-5 stage-A base (`nn_x/w5/lane.sh` A_BASE, `--cooldown_mix 0.02`).

Sources read (cited per row): reports/nnx_VERDICT.md, nn_report.md, sol_gpu.md, sol_solvers.md, R7_ARCH_HPARAM.md,
R6_PLAN.md, r6_recap.md (variation map, matched comparisons), CURRENT_STATUS.md Log 11-12, FEATURES_REVIEW.md,
nn_x/audit/features/findings.txt, INSIGHTS_1112.md, COVERAGE_MAP.md, EXPERIMENT_REGISTRY.md, nn_x/r6/recap/runs.csv
(285 train segments), nn_x/r6/logs/r6_eval_screen_a4u_wd01_cset_typed_adrop0_a4w.txt, nn_x/r7/logs/eval_screen.txt,
nn_x/w5/logs/{w5_eval_B*.txt, w5_e10_base_s95*.json, FROZEN_w5.txt, RUNNING_w5.txt}, the code (nn_x/w5/{train,model,
data}.py, lane.sh, w5_eval.py, tests/test_w5.py) and the raw data (data/competition/val/impressions).

## 0. The recipe under study, and the reference numbers

**10-epoch member (= wave-5 E_BASE = F7 m02 recipe + hygiene; nn_x/w5/lane.sh):**
`--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout
--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --epochs 10 --eval_every 2500 --muon --sched wsd --cooldown_frac 0.2
--aux_anneal 0.1 --retr_m 16 --attn_gate --cooldown_mix 0.02 --lr 2e-3 --save_steps 70000,77500` + HYG
`--save_last --save_holdout --select_min_step 12000 --cliplog 50`. Defaults that are part of the recipe: d 128, heads 4,
blocks 2, hist_layers 1, ff 2, drop 0.1, wd 0.02, warmup 500, cand_budget 32768 / max_feeds 512, max_hist 256, arch tf,
pool = train feeds with p_random > 0.5 and M >= 50 minus the holdout (134,118 feeds), Muon momentum 0.95, AdamW betas
(0.9, 0.98), clip 1.0, lr floor 3%.

**2-epoch screen base (= wave-5 A_BASE = r7 BASE2 with `--cooldown_mix 0.02`):** the same flags with `--lr 1e-3
--epochs 2 --eval_every 1000 --no_test` + HYG, seeds 23 / 24. **This base has never been run**: wave 5 launched only the
stage-B (CD re-cool) and stage-E (10-epoch) jobs (nn_x/w5/logs has no w5_base_s23/s24, FROZEN_w5.txt lists no stage-A
run). The nearest run bases are r7_base_s23/s24 (BASE2 with mix 0.1: val 0.39408 / 0.39334, holdout 0.31779 / 0.31749;
19,670 steps, cooldown from 15,736; A/A same-seed rerun val +0.00003 / holdout -0.00006). The abl screen therefore
needs its own 2 base runs.

**Reference strengths of the current recipe (train-only, holdout-selected):**
- wave-5 10-epoch base runs w5_e10_base_s951-954 (selected @97500 / 95000): val 0.39632 / 0.39691 / 0.39745 / 0.39549,
  holdout 0.32083 / 0.32000 / 0.32026 / 0.32005 (the highest single-member holdouts so far; CURRENT_STATUS 22:54).
- F7x (m02 re-cool of the F7 roots, 7 members): single 0.39667 / 0.31999, 6-mix 0.40055 / 0.32338 (EXPERIMENT_REGISTRY).
- Best pool G3: val 0.40032 / holdout 0.32379 / public 0.392820. Public #1 0.398256 (~+0.0054, a genuinely better model).
- Screen statistics: 2-epoch paired SE_cal (2 seeds) 0.00048-0.00061 val, 0.00025-0.00034 holdout; smallest detectable
  effect ~+0.0008 val (R7_ARCH_HPARAM §0, §3). Single 10-epoch member sd 0.0008-0.0011 val (COVERAGE_MAP statistics).

**Column conventions of the ledger.** "Δ" = paired per-feed delta of the component ON minus OFF (so a positive number is
the component's worth), val first, holdout second, unless the row says otherwise. "Recipe" says what the measurement was
made on: CURRENT (the live Muon+WSD+retr16+pre2s recipe, or its 2-epoch screen form), CURRENT-ish (same lineage, one
setting different, e.g. mix 0.1 instead of 0.02), OLDER (AdamW-cosine era, before retrieval / pre2s / Muon), PROXY (MLP
or GBDT proxy, not the NN). "Gap" = never measured / stale / confounded.

---------------------------------------------------------------------------------------------------------------------

## 1. The ledger

### A. History side

| # | component (current setting) | measured contribution | horizon / seeds / recipe | source | current-recipe measurement? | gap / staleness | ablation flag (code needed?) |
|---|---|---|---|---|---|---|---|
| A1 | **History encoder**: 1 pre-LN transformer layer over [user token, <= 256 newest events]; history token = Linear(item emb) + emb(event type) + emb(event source) + emb(category, shared) + emb(recency index 0..255) + missing-emb vector; candidates cross-attend to it in each of the 2 blocks | `nohist` 0.36808 vs base 0.37361 = **-0.0055 val** (component worth +0.0055); "history target attention matters most (+0.006-0.010)" | 2 ep, 1 seed, OLDER (AdamW cosine, no xt, no retrieval, position tower on) | nn_report "Train-stage ablations", TL;DR | **NEVER** (every one of the 102 Muon+WSD segments has the encoder; runs.csv) | stale by 3 recipe generations; the token inputs (event type / source / category / recency) were never ablated one at a time in any recipe. Low action value: nobody would remove it; its worth now overlaps with retrieval, sim_bias, pre2s and ctx inputs | `--no_hist` exists (also disables retrieval and sim_bias, which need use_hist) |
| A2 | **hist_layers 1** | hl2 (`--hist_extra 1`, zero-init 2nd layer): -0.00059 val (z -1.1) / +0.00007 holdout, M>=50 slice -3.3 SE | 2 ep, seeds 23/24, CURRENT-ish (BASE2 mix 0.1) | R7_ARCH_HPARAM §3 | yes (2 ep) | 10-ep never (deep 3b+2hl ≈ tf at 3/5/10 ep in the AdamW cvnd era: ±0.0002, r6_recap §3.5) | closed |
| A3 | **GRU variant** (`--arch gru`) | F10 single 0.39481 / 0.31885 vs F7 0.39460 / 0.31911; 6-mix 0.39881 / 0.32254 vs 0.39846 / 0.32263; curve-matched +0.0003..+0.0008 val | 10 ep, 6 seeds each, CURRENT minus m02 (F13g = GRU + m02 exists) | EXPERIMENT_REGISTRY; r6_recap §3.5 | yes | none; same strength, no pool gain (cross-family corr 0.92-0.94) | closed (family exists) |
| A4 | **Target-action key view** (`--typed_cross`: last cross block sees user token + event_type 1 / source 1 only) | +0.00043 val (z 1.0) / +0.00020 holdout; FAIL | 2 ep, seeds 23/24, CURRENT-ish | r6_eval_screen_*.txt | yes | none | closed |
| A5 | **History length 256 (`--max_hist 256`)** | inference-time crop 64/128/192/256 → 0.3754 / 0.3782 / 0.3794 / 0.3799 (OLDER, s2_nopos_aux_sim); crop TTA {256,192,128,64} ensemble -0.00100 ± 0.00045 (CURRENT F7/F6 members); "max_hist 256 is binding" | OLDER (inference crop) + CURRENT (TTA) | nn_report K10; R6_PLAN §3, r6_tta §5.1 | partly (crop at inference on current members) | **hist 512 is INFEASIBLE: 256 is the organisers' cap.** Verified here on the raw parquet: 3,528 val feeds from 16 part files, max raw history length 256, 28.8% of feeds at the cap, median 187 (data/flat val/test: max H 256, 29% at cap). Training at a shorter length was never run (no reason to) | n/a |

### B. Candidate side (set attention, depth, width, block form)

| # | component | measured contribution | horizon / recipe | source | current? | gap | flag |
|---|---|---|---|---|---|---|---|
| B1 | **Candidate-set self-attention** (one set block after each of the 2 cross blocks; no positional input) | `noset` 0.37386 vs base 0.37361: **neutral (-0.0002)**; set attention "neutral, with no similarity prior" | 2 ep, 1 seed, OLDER | nn_report ablations; R6_PLAN §1.2 | **NEVER** | stale: the model now carries 34 within-feed-relative inputs (xt_rel, f1..f5 z/pct) that the set blocks could exploit, and INSIGHTS §1 attributes "within-category ordering" to the set attention without an ablation. cset (cosine prior + candidate retrieval `--cretr_m 8 --set_sim_bias`): 2 ep +0.00028 / +0.00090 (FAMILY-CANDIDATE), 10 ep +0.00017 / -0.00001 (warm -0.0033 / cold +0.0086 trade) | `--no_set` exists; also a cost lever (noset ran 13.0 vs 18.4 min in round 1) |
| B2 | **blocks 2** | blocks 3 (`aux_sim_b3`, seed 7) 0.37806 vs aux_sim seeds 2/4/5/9 0.3780-0.3786: **-0.0006, UNPAIRED**; "deep" (3 blocks + 2 hist layers) ±0.0002 at 3 / 5 / 10 ep | 2 ep OLDER (AdamW, no xt); 3-10 ep OLDER (AdamW cvnd) | nn_report round 1; r6_recap §3.5 | **NEVER under Muon** (runs.csv: blocks 2 in 102/102 Muon segments; r7 screened hl2 only and listed "blocks 3: closed earlier" without a re-test) | **STALE** → qualifies for a re-test by the task's rule. Prior low: r7's 9 block-form arms all cost M>=50 feeds, d192 memorises more | `--blocks 3` exists (cost ~1.3x) |
| B3 | **d 128 / 4 heads** | heads 8: -0.0011 (unpaired, round 1); d256/8h: -0.0008 at 2 ep (big_nopos), -0.0004 holdout at 10 ep cosine; **d192/6h under Muon (F12): single 0.39371 / 0.31843 vs F7 0.39460 / 0.31911**, end-of-run decline 3-5x larger (-0.0096) | OLDER + CURRENT (F12, 10 ep, 4-7 seeds) | nn_report; r6_recap §3.5; R6_PLAN §1.2 | yes (width up) | width down / heads never under Muon; low value | closed |
| B4 | **ff 2** | never varied (283/285 runs) | - | r6_recap §2 | NEVER | "FFN is the least-used sublayer" (r7_research) → low prior | `--ff` exists |
| B5 | **Block form** (pre-LN LN, GELU x2 FFN, plain residuals, standard attention, no tabular encoder) | r7: 0/13 promote; derf -0.00124, periln -0.00111, batch2s -0.00145 harmful; muon_in -0.00101, swiglu -0.00107, ftok -0.00114, diff2 -0.00093, hl2 -0.00059, pbld -0.00057, postln -0.00045, hc4 -0.00022, resdrop0 +0.00030, dwa +0.00064 (holdout -0.00027) | 2 ep, seeds 23/24, CURRENT-ish | R7_ARCH_HPARAM §3 | yes | closed at screen power (~+0.0008) | - |

### C. Retrieval and attention priors

| # | component | measured contribution | horizon / recipe | source | current? | gap | flag |
|---|---|---|---|---|---|---|---|
| C1 | **retr_m 16** (A4: TabR-style retrieval of each candidate's top-16 history items by raw cosine; logit = -‖k_c-k_j‖² + a·S + β[type] + sink; values LN(E_type + E_src + E_rec + T(k_c-k_j)); zero-init output Z) | A4 vs base **+0.00186 val (z 2.4) / +0.00326 holdout (cal z 4.3)**; A1+A4 +0.00273 / +0.00389 (3-seed pooled +0.00278 / +0.00382); after FT OOF +0.00195; dissection: T(k_c-k_j) carries ~90% (meta-only -0.00288 holdout); retr16+gate vs base at 10 ep cosine (to 25k) +0.0028 / +0.0030 | 2 ep seeds 23/24/25 OLDER (nnx base: AdamW cosine, pre-Muon, pre-pre2s); 10 ep OLDER (v12_rg_e10 vs v9_e10, cvnd) | nnx_VERDICT §2.3, §3; r6_recap §3.5 | **NEVER removed on the current recipe** (retr 16 in all 83 retrieval segments of Muon+WSD) | the strongest architecture component, measured only on the pre-pre2s / pre-Muon base; with pre2s (history-encoder scores) and ctx_nd the overlap is unknown → worth probably smaller now. **retr_m 8 / 32 / 64 alone: NEVER** (R6_PLAN table: "retr_m 16 only"). a4w (`--retr_m 32 --retr_prod`): -0.00009 / -0.00002 FAIL, confounded by the product term. a4u (`--retr_uniq --retr_tcnt`, duplicate-free slots: 30-32% of baseline slots repeat an item): 2 ep **+0.00117 (z 2.5)** / -0.00004 PASS → 10 ep +0.00036 ± 0.00068 / +0.00018 PROCEED, not adopted (warm -0.0034 / cold +0.0095; mixed 2-member ensemble -0.00028) | omit `--retr_m 16 --attn_gate` (store_true: must be removed from the base string); `--retr_m 32` exists; `--retr_uniq --retr_tcnt` exist |
| C2 | **attn_gate** (A1: elementwise sigmoid output gate in every attention module) | alone +0.00042 (z 0.9) n.s.; increment over A4 +0.00087 (z 1.8) / +0.00063 holdout; after FT +0.00046; mean of 4 pairs +0.00065 ± 0.0003 | 2 ep OLDER | nnx_VERDICT §2.3 | NEVER removed | small, "keep because cheap and never negative"; low value to re-measure | omit `--attn_gate` |
| C3 | **sim_bias** (raw cosine candidate-vs-user / candidate-vs-history-item added to the cross-attention logits with a learned per-head scale, init linspace(0,8)) | nopos_sim vs nopos **+0.0024 val** (0.37624 vs 0.37385); sim_bias 2 (per-type scale + same-category + recency) neutral at 10k steps | 2 ep, 1 seed, OLDER | nn_report ablations | **NEVER removed** (93/93 from-scratch Muon segments have it) | stale, and the retrieval module now has its own cosine term (a·S) and difference vectors, so the current worth of sim_bias is unknown (could be near 0 or still ~+0.002) | omit `--sim_bias` |

### D. Tokens / priors / position

| # | component | measured contribution | horizon / recipe | source | current? | gap | flag |
|---|---|---|---|---|---|---|---|
| D1 | **user0** (K5: is_user0, is_user0·log(1+H) dense features; user_id 0 = pooled bucket, 658 val feeds) | K2+K5 together "neutral in train-only" (v2_batch 0.37804 vs stage-1 seeds 0.3780-0.3786, unpaired) | 2 ep OLDER | nn_report v2 | NEVER removed | cannot be removed with xt inputs on: train.py asserts `--xt requires --user0` (dense layout) → code change; low value | needs code |
| D2 | **domain token** (K2: 3-way feed-domain embedding (production / random-like / exploration) added to every candidate token and again before the head; train feeds = production soft-mixed toward random-like by p_random for M>=50, **M<50 feeds are hard-labelled production (p=0)**; inference = random-like; FT switches to exploration) | neutral (same unpaired comparison as D1) | 2 ep OLDER | nn_report v2; INSIGHTS 3.1 item 7 ("judged only as neutral") | **NEVER removed on the current recipe** | DOMAIN-DIAG (re-score snapshots under the production token; nn_x/insights/data_usage/gpu_domain_diag.py) written, never run; variants never run: soft p for M<50 (INSIGHTS L6), token inside history tokens, no head re-injection, domain-balanced loss (findings P3, low prior). The TRUNC E readout attributes part of the production-artefact absorption to "K9 aux + domain token" without isolating it | omit `--domain` (p_random still loads via `--holdout`, so the pool works) |
| D3 | **no_pos** (position tower off) | base (tower) vs nopos -0.0002; rev_aux (end-only tower) vs nopos_aux -0.0010; PAL tower on the MLP proxy -0.0006 val / +0.0002 holdout; **TRUNC-PROD (true rank-from-end offset, production rows, training only): member -0.00048 ± 0.00051 / -0.00009 FAIL (4 x 10-ep pairs, CURRENT)**; cooldown forms truncp -0.00009 / -0.00013, truncp2 -0.00052 / -0.00030 | OLDER (towers) + CURRENT (TRUNC) | nn_report; sol_solvers §2.1; CURRENT_STATUS 22:54, 22:40 | yes | closed ("the NN's full input set + K9 aux + domain token already absorb the production truncation artefact") | - |

### E. Inputs

| # | component | measured contribution | horizon / recipe | source | current? | gap | flag |
|---|---|---|---|---|---|---|---|
| E1 | **xt preset cvnd_pre2s (34 train-stage per-candidate inputs)** = popularity 8 (seq_cv_invd20, seq_cv_pop, seq_cv_norm, it_hist_{all,et1,et2,et3,et4}_luo) + ctx_nd 10 (hcent, knn_f1, knn_h, knn_h_dev, knn_u, knn_u_dev, max, top5h, top5u, ucent) + ctx_hn 6 + ctx_hs 4 + pre2 6 (pre2_fe_z, pre2_new_z, pre2_fe_s, pre2_new_s, pre2_hmax32, has:pre2_fe) | (i) cv (16 co-vis / LUO cols) vs none: **+0.0032 val / +0.0092 holdout** (v3_cv vs v2_feed_k9, paired seed 23). (ii) cvnd (+20 ctx cols) vs cvraw: **+0.0000 val / +0.0014 holdout** (v8_nd_q15 vs v7_q15; q0: +0.0005 / +0.0017). (iii) pre2s vs cvnd: **+0.00191 ± 0.00058 val (z 3.3; M<50 +0.0040, M>=50 +0.0005) / +0.0002 holdout** at 3 ep (2 seeds); at 10 ep F9 vs F6 +0.0009 val (M<50 +0.0020, M>=50 +0.0001) / +0.0005 holdout (z_run 2.6). (iv) Reliance replays on CURRENT members (r6_cd_m02_r101 / v23_gru_s501; zeroed / permuted within feed): popularity -0.0057 / -0.0151 (zeroing trades coldpos +0.037 vs warmpos -0.024), ctx_nd -0.0110 / -0.0070, ctx_hs -0.0032 / -0.0050, ctx_hn -0.0015 / -0.0014, pre2 -0.0019 / -0.0046; per column it_hist_all_luo -0.0114, seq_cv_pop -0.0055, it_hist_et1_luo -0.0039, seq_cv_norm 0.0000 (redundant with invd20, rho 0.9998). (v) Private-like coverage cost: st_pop -0.00124 / -0.00145 (z -2.2 / -2.6), st_pre2 ±0.0001; pool -0.00175 ± 0.00053 (private-mix -0.00227) | (i)-(ii) 2 ep OLDER (AdamW); (iii) 3 ep OLDER-ish (AdamW cvnd v8) and 10 ep CURRENT-ish (Muon lr 1e-3 mix 0.1, 6-vs-6 families); (iv)-(v) CPU replays of CURRENT members (no retraining) | nn_report round 2, W2/FR-1; r6_recap §3.4; findings.txt §3-4 | pre2: yes (10-ep family pair); popularity / ctx_nd / ctx_hn / ctx_hs: **retrain-without NEVER on any recipe except the old cvraw-vs-cvnd pair** | A zero/permute replay measures reliance of a model trained WITH the inputs, not what a model trained WITHOUT them would score (the old cvnd pair shows the two can differ by an order of magnitude: replay ctx -0.011 vs retrain +0.0000 val). `has:pre2_fe` is a pure early-September date flag on train (26% NaN rows), constant on val/test (findings P4, never tested) | `--xt` accepts a comma list, but XT_REL / XT_RELONLY are keyed by preset name → a custom list silently loses the rank/z channels (confound). **Needs a small flag `--xt_exclude <names>`** that filters the preset list before the XT_REL lookups (the existing `.index`/`in xt_names` logic then adapts) |
| E2 | **xt_rel** (within-feed pct-rank + z-score computed in-model for seq_cv_invd20, it_hist_all_luo, pre2_fe_s, pre2_new_s, pre2_hmax32; the three pre2 raw columns enter ONLY through rank/z) | never ablated. Only evidence: FR-1 fix (clip-safe pre2 raw columns so the rank channel is not tied): cvnd_pre2s vs broken cvnd_pre2 **+0.0007 ± 0.0005 val / +0.0000** (3 ep, 2 seeds) | 3 ep OLDER-ish | nn_report 18:15; FEATURES_REVIEW §2.1 | NEVER | **xt_rel on more inputs NEVER** (seq_cv_pop, it_hist_et1_luo, ctx_nd_knn_h, ctx_nd_ucent, ctx_hs_* are rank-free although they are among the most-relied-on columns). Within-feed ranks are split-invariant by construction, so they are also the one input form immune to the val->private coverage shift (findings §1b: it_hist_all_luo cAUC 0.461, seq_cv_pop 0.470 marginal shift) | needs a small flag `--xt_rel_extra <names>` (append to XT_REL[preset]; n_dense grows by 2 per name) |
| E3 | **f1..f5** (the organisers' production-encoder scores: raw + within-feed z + pct = 15 dense columns, also in the linear "wide" term) | **never ablated and never replayed**. Univariate: f1 is the strongest single input (within-feed slope 0.448 val; "best single feature NDCG 0.349"); f1 / f3 / f5 slopes are 1.14-1.23x STRONGER on production train feeds than on exploration feeds (production-ranker confound) | - | findings.txt §1c; nn_report | NEVER | unknown reliance; the actionable question is the confound, not removal. Cheapest read: zero / permute replay on an existing checkpoint (GPU minutes, no training) | retrain drop needs a flag (`--drop_f`); replay needs only a hook in dense_features |
| E4 | **Other in-model dense features (61)**: cos/dot with user, candidate-vs-history cosine pools (max all / last 8 / last 32, mean, soft-max, top-3, per event type), category affinity (log count, share, per-type counts, recency of first same-category event), feed category share, log M, log H, within-feed z / pct of the similarity features | never ablated individually; the h_ (history-structure) and rc_ (within-category relative) feature probes over G3 are null ("the in-model category-affinity block already covers this") | CPU probes on G3 | findings.txt §5 | NEVER | low value | - |
| E5 | xt_drop 0 (ID-column input dropout) | 0.15 / 0.3 negative; age-conditioned dropout stale-val +0.0009 but holdout -0.0019 | OLDER | CURRENT_STATUS 3d; ROUND5 §2a | - | closed | - |

### F. Losses and auxiliary task

| # | component | measured contribution | horizon / recipe | source | current? | gap | flag |
|---|---|---|---|---|---|---|---|
| F1 | **Listwise softmax CE** (target y/K per feed) | BCE -0.0009 (0.37272 vs 0.37361); loss zoo on an MLP proxy: 14 alternatives, none better (LS +0.00007, focal +0.00011, LambdaLoss +0.00006, PL-over-positives -0.00006; RankNet -0.00033 … ListMLE -0.00516); CE's negative-gradient allocation ≈ LambdaRank (top-5 32% vs 35%; same-category 39.0% vs 40.2%); Spearman(NDCG, -CE) 0.95 over 152 members; every feed re-weighting null or negative (Optuna nested -0.0010 ± 0.0011) | 2 ep OLDER (NN); PROXY; CURRENT members (gradient analysis) | nn_report; sol_solvers §2.1; INSIGHTS 6 and §4 | yes (analysis) | closed, no ablation needed | `--loss bce` exists |
| F2 | **loss_norm feed** (K1b: equal per-feed weight instead of the per-batch candidate mean, which gives an M~300 feed ~4.7x the weight of an M<=64 feed) | **+0.0020 val / +0.0006 holdout** (v2_feed vs v2_batch, paired seed 21); idcg weighting ≈ feed | 2 ep OLDER (AdamW, no xt, MSE aux) | nn_report v2 | **NEVER re-measured** (3 F6cd continuations accidentally ran with 'batch', unpaired) | stale; the K9 aux is still batch-normalised while the main loss is per feed (aux:main ratio rises 4.3x with M; AUXF feed-norm was null in both forms: CD -0.00010 / -0.00012) | `--loss_norm batch` exists (last occurrence wins) |
| F3 | **K9 aux** (second head, listwise CE toward softmax(-pos/10) on production feeds, weight 0.5·(1-p_random), annealed to 0.1 with the cooldown lr, batch-normalised) | MSE aux vs none +0.0020 (nopos_aux vs nopos, 1 seed); **listwise vs MSE +0.0024 val / +0.0027 holdout** (v2_k9 vs v2_batch, paired). Variants on the 2-ep screen base (mix 0.1): **tau 30 -0.00091 (holdout -0.00109, z -3.9) WORSE**; PL top-20 +0.00059 (z 1.3) NULL (M<50 +0.0024, M>=50 -0.0006; gain vanishes under private-like coverage); PL + aux_pos 1.0 +0.00051 NULL (vs PL -0.00008); aa5 (anneal to 0.5, CD) STOP -0.00007; auxf (feed-normalised, CD) -0.00010 / -0.00012 STOP; category-intent head -0.0014 | OLDER (worth); CURRENT-ish (variants, 2 ep seeds 23/24; CD roots) | nn_report v2; sol_gpu; CURRENT_STATUS 12:39, 22:03 | variants yes; **total worth (aux_pos 0) NEVER on any recipe with retrieval / pre2s / Muon** | the aux is credited (with the domain token) for absorbing the production truncation artefact, and K9 is "the trunk regulariser"; its current worth is unknown. Never run: `--aux_pos 0`; `--aux_pos 1.0` with listwise alone; `--aux_anneal 0` (anneal to zero, the opposite of aa5) | `--aux_pos 0` exists (the aux head is then not built); `--aux_anneal`, `--aux_pos` exist |

### G. Optimiser, schedule, cooldown, regularisation, batch, epochs

| # | component | measured contribution | horizon / recipe | source | current? | gap | flag |
|---|---|---|---|---|---|---|---|
| G1 | **Muon** (RMS-matched, every 2-D Linear incl. gate / retrieval matrices; embeddings, LN, biases, 1-output layers on AdamW) | vs AdamW **+0.00201 val (z 3.4) / +0.00295 holdout (cal z 4.2)**; 10 ep cosine (to 30k) +0.0012 / +0.0010; muon_in (input projections on AdamW) -0.00101 / -0.00031 | 2 ep seeds 23/24 OLDER; 10 ep OLDER; muon_in CURRENT-ish | nnx_VERDICT §2.3; r6_recap §3.5; R7 §3 | partly | removal not needed (adopted; nothing suggests a reversal). Momentum 0.95 / betas / clip 1.0 / warmup 500 never varied; CLIPLOG: 0% clipped cooldown steps, so clipping is not binding | omit `--muon` (also drops split k/v) |
| G2 | **WSD schedule** (1-sqrt cooldown to the 3% floor, no stage 2) | WSD vs stage 2 on AdamW +0.00114 (z 2.3) / +0.00074 (cal z 1.1); Muon+WSD vs Muon+stage 2 +0.00158; **stage 2 after WSD -0.00121 / -0.00147 (harmful)**; Muon+WSD vs base+stage 2, 3-seed pooled +0.00306 / +0.00331; cooldown improves the mean function: 7-seed ensemble cooled minus plateau-end +0.0031 ± 0.0006 | 2 ep OLDER; 10 ep CURRENT (r6 recap) | nnx_VERDICT; r6_recap §5 | yes | adopted | `--sched cos` exists |
| G3 | **lr** (2e-3 at 10 ep; 1e-3 on the 2-ep screen) | 10 ep F7 (2e-3) vs F9 (1e-3): **+0.00078 val (z_run 1.4) / +0.00067 holdout (z 2.9)**, halves the late decline; 2 ep WSD: 2e-3 **-0.00097 val** (both seeds) / +0.00056 holdout; lr > 2e-3 never | CURRENT | r6_recap §3.1; nnx_VERDICT §5 | yes | horizon-dependent: the 2-ep screen runs at the lr that loses at 10 ep (screen / recipe mismatch to keep in mind when reading arms that interact with the lr) | `--lr` |
| G4 | **cooldown_frac 0.2** | never varied from scratch; CD wave 3: cf4 (40% at matched compute) **-0.00121 (z -2.3) / -0.00023**, coldpos -0.0058; e3 (3-ep cooldown) -0.00012; s2c2 (12 ep) -0.00011 → all STOP | CURRENT (CD, roots 101-104 vs m02) | CURRENT_STATUS 15:29 | yes (longer) | shorter (10%) never; low prior ("longer cooldowns lose cold positives") | `--cooldown_frac` |
| G5 | **cooldown_mix 0.02 (m02)** | m02 vs c0 (mix 0.1): **+0.00117 ± 0.00039 val (z 3.0) / +0.00134 ± 0.00022 holdout (z 6.0)**, all slices positive (coldpos +0.0045), 7-member ensemble +0.00108 / +0.00137 → PASS-CD, default cooldown. m0 vs m02: val +0.00002, coldpos -0.0023 (z -4.5) → record only. mix 0.3: collapse (-0.027). Cooldown-data arms beyond m02: 0 of 15 (pp02 -0.00074, aug -0.00014, d0 -0.00119, d20 -0.00015, aa5, hcap +0.00007, rec33 -0.00212, rho70 -0.00350, hzcd +0.00026 / -0.00052, truncp, truncp2, auxf, cf4, e3, s2c2) | CURRENT (CD re-cools of the F7 roots @77500, 2-epoch cooldown) | CURRENT_STATUS 10:55-18:47, 22:54 | yes | **dose 0.01 / 0.04 NEVER**. Dose arithmetic: at 10 ep m02 = 0.65 extra pool passes; the 2-ep screen at mix 0.02 = **0.19 passes** (mix 0.1 at 2 ep = 0.93), so the 2-ep base is under-dosed relative to the 10-ep recipe, and a 2-ep screen is nearly blind to the dose (R6_PLAN: "a 2-epoch screen cannot see cooldown-regularisation knobs"; INSIGHTS: m02's gain appears after k~10k of a 19.7k cooldown) | `--cooldown_mix` (CD harness = nn_x/w5 stage B, roots 101-104 vs w5_cd_m02ref, is the right place) |
| G6 | **aux_anneal 0.1** | aa5 (0.5) STOP (-0.00007 / +0.00035) | CURRENT (CD) | CURRENT_STATUS 12:39 | yes (up) | 0 / off never | `--aux_anneal` |
| G7 | **wd 0.02** | w10 (0.1 in the cooldown) -0.00055 val, ensemble holdout -0.00068 → STOP | CURRENT (CD) | EXPERIMENT_REGISTRY | yes (up) | never varied from scratch; down never | `--wd` |
| G8 | **dropout 0.1** (residual / FFN / retrieval + attention probabilities) | resdrop0 (2 ep) +0.00030 / +0.00040 null; d0 (0 in the cooldown) -0.00119 (z -2.4) / -0.00054; d20 -0.00015 / +0.00031; adrop0 -0.00023 (M>=50 -2.4 SE) FAIL | CURRENT-ish / CURRENT | R7 §3, §5; CURRENT_STATUS 15:37; r6 Q2 | yes | closed | `--drop`, `--attn_drop`, `--res_drop` (r7) |
| G9 | **batch 32,768 candidates / 512 feeds** | batch2s (x2, sqrt lr rule) **-0.00145 / -0.00109**, 0.815x time per sample → COST FAIL; GNS B_simple 4.9-9.4K feeds vs 317 per step | CURRENT-ish | R7 §3, §5; r6_recap §4.2 | yes (up) | linear-rule x2 and smaller batches never; low value | `--cand_budget --max_feeds` |
| G10 | **epochs 10** | F9 (10 ep) vs F8 (2 ep): single +0.00060 / +0.00069 (z_run 4.0), **6-seed ensemble +0.0040 / +0.0031** (seed corr 0.978 → 0.938); 12 ep (s2c2) -0.0001; plateau mean function saturated by ~60k (7-seed ens 60k 0.39660 vs 77.5k 0.39551); 5-7 ep under Muon never (P1 cost lever) | CURRENT | r6_recap §3.2, §5; CURRENT_STATUS 15:29 | yes | none that changes the recipe | `--epochs` |
| G11 | **Exploration pool definition** (p_random > 0.5, M >= 50, holdout excluded; 134,118 feeds) | pp02 (p > 0.2, +47% feeds) -0.00074 / +0.00034 STOP; short feeds in the pool -0.00053 (proxy) | CURRENT (CD) / PROXY | CURRENT_STATUS 15:37; sol_solvers §2.6 | yes | closed | `--pool_p` |

### H. Evaluation and checkpoint selection

| # | component | measured contribution | source | current? | gap |
|---|---|---|---|---|---|
| H1 | **30K random-like train holdout (p_random > 0.9, M >= 50, rng 12345; excluded from training and the pool) + `--select_on holdout`** | the only val-label-free selector; its per-feed SE is ~2.3x too small (run-level selection over ~20 evals; nnx_VERDICT §2.2); for end-rising m02 families holdout selection "buys only luck (+0.0002..+0.0008 per member)" (INSIGHTS 13); wave-5 hygiene (select_min_step 12k + last checkpoint kept) produced the highest single holdouts (0.3200-0.3208); same-seed non-determinism \|Δval\| 0.0007 | nnx_VERDICT §2.2; INSIGHTS 13; CURRENT_STATUS 22:54; r6_recap §4.5 | yes | `--select_on val` would leak val into member selection (never used for members; protocol, not a lever). Plateau weight averaging: single +0.0004, 6-member ensemble -0.0025 / -0.0016 (closed); stage-1 EMA null |
| H2 | eval_every 2500 (10 ep) / 1000 (2 ep) | epoch-phase sawtooth: late-epoch evals +0.00025 val / +0.00029 holdout (z 4.3 / 7.0) → compare checkpoints at matched phase | INSIGHTS 13 | yes | none (reading rule) |

### I. Post-training: fine-tune and blend

| # | component | measured contribution | source | current? | gap |
|---|---|---|---|---|---|
| I1 | **Val fine-tune n2cpv3** (n2c_pre_v3 extras, u0_singleton folds, cos + EMA 0.99, 4 ep, lr 5e-5, 5-fold GroupKFold by user) | fta vs const-lr snapshot FT on Muon checkpoints +0.00112 OOF; per member +0.0009..+0.0014 (F7) but **≈ 0 at pool level on pre2s/m02 trunks** (FT raises member correlation 0.945 → 0.957); FT of 3 of 20 members = FT of all 20 (+0.00001 ± 0.00022); Optuna retune CLOSED (+0.00055, z 1.3); M-split fresh confirmation FAIL (-0.00017 ± 0.00031); forward-transfer test +0.00055 ± 0.00042 | nnx_VERDICT; INSIGHTS 10; CURRENT_STATUS 18:54, 21:18, 22:03 | yes | closed ("no more FT work") |
| I2 | **Blend 0.5 FT + 0.5 train-only** (member-equal per-feed z-mean within each branch) | w_ft ≠ 0.5: nested +0.0002 (z 0.96) closed; **public: G3 train-only half alone 0.392346 vs G3 0.392819 (-0.00047)** → the FT half is worth ~+0.0005 forward in time | ROUND5 §2c; CURRENT_STATUS 21:52 | yes (public) | none |
| I3 | **Ensemble of gated families** (G3: F10 + F7x + F7x70, 20 members) | infinite-member headroom ≈ +0.0002; adding any existing family <= 0; stacking / routing / weights closed | INSIGHTS 1; nn_x/stack | yes | saturated: only stronger members move the pool |

### Reading of the ledger

1. **Every large measured contribution was measured on an older recipe**: retrieval (+0.002-0.003), sim_bias (+0.0024), K9 (+0.002 + 0.0024), loss_norm feed (+0.0020), cv/cvnd inputs (+0.003 val / +0.009 holdout), the history encoder (+0.0055) and the domain / user0 tokens (neutral) all date from the AdamW-cosine, pre-retrieval, pre-pre2s era (2 epochs, mostly single seeds). Only m02, the K9 variants, a4u / cset / typed / adrop0 / a4w, the r7 block arms, the cooldown arms, TRUNC and pre2s (10-ep family pair) were measured on the current lineage.
2. **None of the current recipe's components has ever been removed on the current recipe**: retr_m 16, attn_gate, sim_bias, domain, user0, K9 (aux_pos 0), loss_norm feed, the set attention, and the popularity / ctx_nd / ctx_hn / ctx_hs input families (retrain-without) are all NEVER rows. The components overlap (retrieval and sim_bias both read raw cosine; pre2s and ctx_nd both encode history similarity; K9 and the domain token both encode the production regime), so the old deltas are upper bounds for the current worth.
3. **The scale-up space that the ledger leaves open** is narrow: retrieval depth / form (never varied alone), xt_rel on more inputs (never), blocks 3 under Muon (stale, unpaired), domain-token variants (never), m02 dose (never; needs the CD harness). History length is capped by the data (256), width/depth/heads/block form, losses, K9 target forms, cooldown length / dropout / wd / pool composition, batch and FT knobs are closed by current-recipe evidence.
4. **The 2-epoch screen is adequate for removals** (effects >= 0.002 expected, MDE ~0.0008) and for architecture scale-ups (r7 lesson: 2-ep members ≈ the 2-ep mean function), but **blind to cooldown-dose knobs** (0.19 pool passes at mix 0.02); those go to the stage-B CD harness.

---------------------------------------------------------------------------------------------------------------------

## 2. Gap summary (never measured on the current recipe, or stale)

| gap | kind | best existing number (recipe) | why it matters now |
|---|---|---|---|
| K9 aux removed (`--aux_pos 0`) | NEVER anywhere with retrieval / pre2s / Muon | +0.002 (MSE vs none) and +0.0024 (listwise vs MSE), OLDER | the "trunk regulariser"; also credited for absorbing the production artefact (TRUNC FAIL). Its worth decides whether more aux work is worthwhile at all |
| retrieval + gate removed | NEVER on current | +0.0027 / +0.0039 (OLDER 2 ep) | the strongest architecture item; its current worth bounds what retrieval scale-ups can add |
| sim_bias removed | NEVER on current | +0.0024 (OLDER, 1 seed) | overlaps with the retrieval's cosine term; may be redundant now |
| domain token removed | NEVER on current (neutral, unpaired, OLDER) | - | DOMAIN-DIAG never run; L3 / L6 variants are gated on it |
| loss_norm batch | NEVER re-measured | +0.0020 (OLDER) | K1 is the second-largest loss-side win; the aux/main normalisation mismatch (AUXF null) makes its current size uncertain |
| set attention removed | NEVER on current (neutral, OLDER) | -0.0002 | a free cost lever (~-25% step time) if still neutral; a load-bearing finding would motivate cset-type scale-ups |
| popularity / ctx_nd / ctx_hn / ctx_hs families retrained without | NEVER (only CPU replays of trained members) | replays -0.006 / -0.011 / -0.0015 / -0.003 (CURRENT members) | replay ≠ retrain (old cvnd pair: +0.0000 val retrain vs -0.011 replay); popularity is the ONLY private-shifted family (st_pop -0.0012..-0.0023 private-mix) |
| f1..f5 reliance | NEVER (no replay, no retrain) | - | production-confounded inputs; reliance unknown |
| xt_rel removed / extended | NEVER | +0.0007 (FR-1 fix, 3 ep) | rank channels are split-invariant; cheap to extend |
| retr_m != 16 alone | NEVER (a4w confounded) | a4u +0.0012 → +0.0004 (10 ep) | depth is the obvious lever on the strongest component |
| blocks 3 under Muon | NEVER (stale, unpaired -0.0006) | - | the task's "only if stale" condition holds |
| m02 dose 0.01 / 0.04 | NEVER | m0 ≈ m02 on val, mix 0.1 -0.0012 | bracket the only adopted cooldown change; CD harness |
| hist 512 | infeasible | raw cap 256 (verified) | closed |

---------------------------------------------------------------------------------------------------------------------

## 3. Scale-up candidates for the components that carry the most gain

| component (measured gain) | candidate | status / feasibility | expected (2 ep) | code |
|---|---|---|---|---|
| Retrieval (+0.002-0.003) | **retr_m 32 alone** | never run alone (a4w had the product term) | -0.0003..+0.0008; cost ~1.2x, +2-3 GB | flag exists |
| | **retr_m 32 + duplicate-free slots + per-type counts** (`--retr_uniq --retr_tcnt`) | a4u at 16 passed the 2-ep screen (+0.0012); at 32 the dedup makes the extra depth count (30% of baseline slots are repeats) | 0..+0.0012; warm/cold trade expected (a4u pattern) | flags exist |
| | retr_m 64 | memory +4-6 GB, time ~1.5x; only if 32 >= 0 | bracket | flag exists |
| | retrieval over the candidate's **same-category history items** (second HistRetrieval with S masked to same category, zero-init Z) | new mechanism; prior low-moderate: the dense block already holds category affinity (counts, share, per-type counts, recency) and the h_ / rc_ probes over G3 were null, but item-level same-category evidence (which items, their event types, difference vectors) is not available to the model today; INSIGHTS: 53% of the loss is the category choice and 37% is "right category, buried" | 0..+0.0008 | **new code** (~40 lines + tests), second wave |
| | key dim dk 64 → 128 / wider T(k_c-k_j) MLP | never varied; "T carries 90% of the gain" | 0..+0.0005 | new code (small) |
| K9 aux (+0.002 / +0.0024) | stronger weight (aux_pos 1.0, listwise alone) | never alone (only with PL: -0.0001 vs PL) | ~0 (prior from PL+w1) | flag exists |
| | per-position targets | = PL top-20: NULL (+0.0006, z 1.3; M<50 gain vanishes under private-like coverage); tau 30 WORSE; aa5 STOP; auxf NULL | closed | - |
| | anneal to 0 (`--aux_anneal 0`) | never; horizon-dependent (acts in the cooldown) | ~0 at 2 ep; CD harness if at all | flag exists |
| Inputs (+0.003-0.005) | **xt_rel on more inputs** (`--xt_rel_extra seq_cv_pop,it_hist_et1_luo,ctx_nd_knn_h,ctx_nd_ucent`) | never; the rank channel was worth +0.0007 on pre2 (FR-1); split-invariant → also a private-robustness lever (read stale-val) | -0.0003..+0.0008 | **new code** (small: append to XT_REL) |
| | more input families | the on-disk and 52 new feature families are null over G3 (findings §5-7) | closed | - |
| Candidate-set attention (neutral, stale) | **blocks 3** | stale (round-1 unpaired, AdamW) | -0.0005..+0.0005; cost 1.3x | flag exists |
| Domain token (neutral, stale) | soft p_random for M<50 feeds (L6; today p = 0 → every M<50 feed is hard-labelled production for the token and the aux, although ~214K exploration feeds hide among them) | never; needs a per-M-bucket prior or a constant p (the detector is unidentifiable for M<50) | 0..+0.0004 (M<50 slice) | new code (small), second wave |
| | DOMAIN-DIAG (re-score the snapshots under the production token) | script written (nn_x/insights/data_usage/gpu_domain_diag.py), never run; GPU minutes; gates L3 | decision value only | exists |
| m02 (+0.0012) | **dose 0.01 / 0.04** | never; at 2 ep the pool dose is 0.19 passes → run as CD arms (stage B, roots 101-104 vs w5_cd_m02ref_r<r>, CDSEED 900+root) | m04 ~0..-0.0005 (between m02 and mix 0.1's -0.0012); m01 ~0 (m0 ≈ m02 on val, coldpos -0.0023) | flag exists |
| History (+0.0055) | hist 512 | **infeasible** (raw cap 256) | - | - |
| Muon / WSD / epochs | - | lr > 2e-3, > 10 ep, longer cooldown, batch x2: all closed | - | - |

---------------------------------------------------------------------------------------------------------------------

## 4. Ranked screen arms (<= 12) on the 2-epoch base

**Base (both seeds REQUIRED, the stage-A base never ran):** in the new harness copy `nn_x/abl/` (copy of nn_x/w5: it has
the HYG flags, `w5_eval.py A` paired readout, `stale_pred.py`, `tests/test_w5.py` AST + e2e bit-identity proofs; set the
lane allow-list to GPUs N and 7 (lane.sh `ALLOWED` default is "<ids>"; common.py accepts 0-7), variant prefix `abl_`,
outputs under nn_x/abl/{ckpt,logs,preds}):

```
CUDA_VISIBLE_DEVICES=<5|7> OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 \
<repo>/.venv/bin/python train.py --variant abl_base_s<23|24> \
  --no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 \
  --select_on holdout --loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 1e-3 --sched wsd --cooldown_frac 0.2 \
  --cooldown_mix 0.02 --aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2 --eval_every 1000 --no_test \
  --save_last --save_holdout --select_min_step 12000 --cliplog 50 --seed <23|24>
```

Arm flags: value flags are APPENDED last (argparse: last occurrence wins); store_true flags (`--sim_bias --domain
--retr_m 16 --attn_gate`) cannot be switched off by appending, so REMOVAL arms must drop the flag from the base string
(the abl lane should build the base from a list and let an arm spec name flags to drop). Each arm: seeds 23 and 24,
paired per feed against abl_base of the same seed. 2 base + 12 arms x 2 seeds = 26 runs of ~30-50 min (3 per GPU, 2
GPUs → 6 lanes, ~3.5-4 h wall).

| rank | arm | type | exact change vs the base | new code? | mechanism / what it tells us | expected 2-ep val Δ (arm - base) | cost / step |
|---|---|---|---|---|---|---|---|
| 1 | **no_aux** | ABLATION | append `--aux_pos 0` (the aux head is not built; `--aux_anneal` becomes inert) | no | total worth of the K9 production-order distillation on the current recipe (never measured anywhere with retrieval / pre2s / Muon; OLDER evidence +0.002 + 0.0024). Decides whether aux variants deserve any more GPU, and tests the TRUNC-FAIL explanation ("K9 + domain token absorb the production artefact"): watch the M<50 / production-heavy slices | **-0.002..-0.005** (expect negative) | ~0.98x |
| 2 | **no_retr** | ABLATION | drop `--retr_m 16 --attn_gate` from the base | no | worth of retrieval + gate on the current inputs (OLDER: +0.0027 / +0.0039). If the worth has shrunk toward +0.001, pre2s / ctx_nd already supply what retrieval gave and retrieval scale-ups (ranks 3-4) have little room; if it is still >= +0.002, depth is the lever | **-0.0015..-0.003** | ~0.85x |
| 3 | **retr32** | SCALE-UP | append `--retr_m 32` | no | retrieval depth alone (never run alone; a4w was confounded by the product term). Keys / values unchanged, twice the retrieved slots | -0.0003..+0.0008 | ~1.2x, +2-3 GB |
| 4 | **retr32u** | SCALE-UP | append `--retr_m 32 --retr_uniq --retr_tcnt` | no | depth 32 over DISTINCT items with per-type counts (a4u at 16: 2-ep +0.00117 z 2.5 PASS, 10-ep +0.00036; 30-32% of baseline slots repeat an item, so depth without dedup is partly wasted). retr32u - retr32 isolates the dedup at depth 32. Known risk: a4u-type arms trade warm for cold positives (read coldpos / warmpos, pmix) | 0..+0.0012 | ~1.3x (fp32 dedup scores) |
| 5 | **xdrop_pop** | ABLATION | append `--xt_exclude seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo` (XT_REL then keeps only the pre2 columns) | **yes, small**: filter the preset list before the XT_REL / XT_RELONLY / xt_idcols lookups; default '' = identical | retrain without the 8 popularity columns (replay reliance -0.006 / -0.015; the ONLY private-shifted family: st_pop -0.0012..-0.0015 member, -0.0023 private-mix pool). Readout adds the stale-val replay: a model without popularity inputs cannot lose from the coverage shift, so val Δ vs stale-val Δ prices the private trade directly | **-0.002..-0.005 val**; stale-val Δ ≈ val Δ + 0.0012 | ~1.0x |
| 6 | **xdrop_ctxnd** | ABLATION | append `--xt_exclude ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_h,ctx_nd_knn_h_dev,ctx_nd_knn_u,ctx_nd_knn_u_dev,ctx_nd_max,ctx_nd_top5h,ctx_nd_top5u,ctx_nd_ucent` | same flag | retrain without the most-relied-on family (replay -0.011 / -0.007; the only retrain pair, cvraw vs cvnd in the OLDER recipe, was +0.0000 val / +0.0014 holdout for ctx_nd + hn + hs together). Tells whether the model's dependence on feed-internal neighbourhood density is real worth or redundant with retrieval / set attention | -0.0005..-0.003 | ~1.0x |
| 7 | **no_sim** | ABLATION | drop `--sim_bias` from the base | no | worth of the cosine attention prior now that the retrieval module has its own cosine term a·S and difference vectors (OLDER +0.0024, 1 seed). If ~0: a free simplification and a sign that attention priors are saturated; if still >= +0.0015: priors are load-bearing and cset-type priors deserve a second look | -0.0005..-0.0025 | ~0.98x |
| 8 | **no_domain** | ABLATION | drop `--domain` from the base (p_random still loads through `--holdout`; the pool and the K9 (1-p) weight are unchanged) | no | worth of the traffic-domain token under m02 + K9 (OLDER: neutral, unpaired). Gates INSIGHTS L3 / L6 (exam offsets, soft p for M<50) and the production-confound story (findings P3) | -0.001..+0.0005 | 1.0x |
| 9 | **xrel_more** | SCALE-UP | append `--xt_rel_extra seq_cv_pop,it_hist_et1_luo,ctx_nd_knn_h,ctx_nd_ucent` (+8 dense columns) | **yes, small**: extend XT_REL[preset]; n_dense follows automatically; default '' = identical | within-feed rank / z channels for the four most-relied-on rank-free inputs (FR-1: the rank channel was worth +0.0007 on pre2). Ranks are split-invariant, so read stale-val too: a gain that is larger under private-like coverage is a private lever | -0.0003..+0.0008 | 1.0x |
| 10 | **lnorm_batch** | ABLATION | append `--loss_norm batch` | no | worth of K1 (equal per-feed weight; OLDER +0.0020 / +0.0006) on the current recipe, where the aux is still batch-normalised. Read M<50 vs M>=50 (K1 moves weight to short feeds) | -0.001..-0.0025 | 1.0x |
| 11 | **blocks3** | SCALE-UP | append `--blocks 3` (a third cross + set block pair) | no | the only capacity knob whose evidence is stale (round-1 unpaired -0.0006, AdamW, no xt; never under Muon). Prior low (r7: every block-form arm cost M>=50; d192 memorises more), but the task's re-test condition holds | -0.0005..+0.0005 | ~1.3x |
| 12 | **no_set** | ABLATION | append `--no_set` | no | worth of the candidate-set self-attention on the current recipe (OLDER: neutral). If still neutral it is a ~25% step-time cost lever for every future family and a simplification; if load-bearing (>= +0.001) it says within-feed interactions matter and cset-type priors are worth revisiting at 10 ep | -0.001..+0.0003 | ~0.75x |

**Not in the 12, and why (second wave or other harness):**
- m01 / m04 (`--cooldown_mix 0.01 / 0.04`): run in the stage-B CD harness (roots 101-104 vs w5_cd_m02ref_r<r>, `--init
  nn/ckpt/v17_pre2s_s<r>@77500.pt`, B_BASE + HYG, CDSEED 900+root), not in the 2-ep screen (0.19 pool passes; blind).
- retr_cat (same-category retrieval), dom_soft (L6), retr_dk128, retr64: new code or conditional on ranks 3-4 / 8.
- xdrop_pre2: a current-lineage 10-ep pair exists (F9 vs F6 +0.0009 / +0.0005; 3 ep +0.0019 z 3.3); lowest information.
- xdrop_ctxhn_hs (10 columns): replay reliance -0.0015 / -0.003; run if xdrop_ctxnd is load-bearing.
- f1..f5: read by zero / permute REPLAY on an existing checkpoint (GPU minutes, no training); a retrain drop has no action.
- no_gate alone, no_user0 (needs code: `--xt requires --user0`), aux_pos 1.0 listwise, aux_anneal 0: low prior / horizon-only.
- Muon → AdamW, WSD → cosine, lr, epochs, dropout, wd, batch, cooldown length, pool composition, block form: closed by
  current-recipe evidence (ledger §G, §B5).

**Proposed readout (to be pre-declared with `date -u` in nn_x/abl before the first launch; the run step owns it):**
- Harness: `w5_eval.py A`-style paired per-feed NDCG (src/metrics.ndcg_per_feed) of the train-only predictions, arm
  minus abl_base on the same seed, pooled over seeds 23/24; primary = the LAST checkpoint (k = 19,670; the holdout is
  then an unselected judge), secondary = the selected checkpoint (holdout-best at k >= 12k). SE_cal = sqrt(SE_feed² +
  sigma_run²/2) with sigma_run 0.00035 val / 0.00015 holdout (r7 A/A). Slices: M<50, M>=50, Tue, Thu, coldpos,
  warmpos; holdout all + late 20%; the 08_levers per-class table; train minutes vs base (cost levers); for the input arms
  (xdrop_pop, xdrop_ctxnd, xrel_more) also stale-val ('both_pre2both', stale_pred.py) and pmix.
- ABLATION verdicts: LOAD-BEARING iff pooled val Δ < -2 SE_cal AND holdout Δ < 0 (component worth >= ~0.001);
  DISPENSABLE iff |val Δ| <= 1 SE_cal AND holdout Δ >= -1 SE_cal AND no slice < -2 SE_feed (candidate simplification /
  cost lever; confirm with one 10-ep pair before adopting); otherwise INCONCLUSIVE (report the sign).
- SCALE-UP verdicts (r7 stage-1 rule): PROMOTE iff pooled val Δ > 1.5 SE_cal AND holdout Δ >= 0 AND (both seeds' val
  Δ > 0 OR both holdout Δ > 0) → 10-epoch pair with E_BASE + HYG (the w5_e10_base_s951-956 runs are exact flags-off
  references if nn_x/abl is proven bit-identical to nn_x/w5), judged by the unified rule (val > 2 SE, holdout >= 0, no
  slice < -2 SE) plus the ensemble-level check; a cold/warm trade (|coldpos - warmpos| > 2 SE) requires the stale-val
  and pmix reads before any adoption.
- Flags-off identity before launch: AST proof (train.py minus the abl statements == the frozen nn_x/w5/train.py,
  md5 3c7667e373d34daf996e3b475a581b10 in nn_x/w5/logs/FROZEN_w5.txt) + end-to-end bit-identity (tests/e2e_w5.py
  pattern) on one GPU; every new flag (`--xt_exclude`, `--xt_rel_extra`) defaults to '' and is guarded.
- GPU rules for the run step: GPUs N and 7 only (both idle at 23:50 UTC: 0 MiB, 0 compute processes), <= 3 of our
  processes per GPU, check nvidia-smi (< 4 compute processes, >= 60 GB free) before every launch, CUDA_VISIBLE_DEVICES =
  one id, 12 CPU threads, PIDs in nn_x/abl/RUNNING.txt (resolve real PIDs with pgrep + exact-args check, never the
  setsid wrapper's $!), stop only our own PIDs.

## 5. Files
- This ledger: <repo>/nn_x/abl/ledger/ledger.md (the only file written by this step).
- Raw-history cap check (CPU, polars, 16 val part files): max raw history length 256, 28.8% of feeds at the cap,
  median 187 → hist 512 infeasible.

---------------------------------------------------------------------------------------------------------------------

## 6. SECOND WAVE pre-declaration (written 2026-11-13T04:05:17Z, BEFORE the first second-wave run; ABLATION_PLAN_1113 §3.3)

Trigger (screen readout logs/abl_eval.txt, 03:46 UTC, frozen rules): no_retr LOAD-BEARING with worth +0.00337 val / +0.00294
holdout (>= +0.002) and retr32 FAIL / flat (-0.00050 / -0.00048) -> row "retr_cat" is due; xdrop_ctxnd LOAD-BEARING
(+0.00135 / +0.00030) -> row "xdrop_ctxhn_hs" is due; the "always" row f1..f5 replay is due. Code state: train.py +
abl_retr.py (new, off-by-default `--retr_cat 0`), abl_lane.sh replaced by mv (ALLOWED "5 7 3", two arms appended),
abl_eval.py ARMS extended (rules untouched), tests/test_abl.py extended (CPU 92/92 PASS at 04:03 UTC; GPU part runs now
on GPU N before the first retr_cat launch). model.py / data.py / common.py / cd_arms* unchanged (byte-identical to w5).

| arm | type | exact change vs the base (A_BASE + HYG, seeds 23 / 24) | mechanism |
|---|---|---|---|
| retr_cat | SCALE-UP | append `--retr_cat 16` (variants abl_retr_cat_s23 / s24) | a SECOND HistRetrieval (16 slots, same parameterisation and init as the base retr_m 16 module: logit -\|\|k_c-k_j\|\|² + a·S + beta[type] + sink, values LN(E_type + E_src + E_rec + T(k_c-k_j)), zero-init output Z) over ONLY the candidate's SAME-CATEGORY valid history items (S masked to -2.0 for the others -> logit -inf, p = 0); a candidate without a same-category item (or without an embedding) gets exactly 0 from the module. Added to the candidate token at the same point as the base retrieval (hook on model.retr). Dropout inside its value MLP draws from a dedicated generator, so flag-ON at init is bit-identical to flag-OFF in eval AND train mode (tests C4 / G2). Parameters built after the Ranker under extra_init (baseline init unchanged), optimised like the base retrieval (wk / t1 / t2 / Z -> Muon). Checkpoint cfg carries retr_cat=16 (load with abl_retr.load_ranker). Expected cost ~1.1-1.2x step time, +2-3 GB |
| xdrop_ctxhn_hs | ABLATION | append `--xt_exclude ctx_hn_top1w,ctx_hn_rrw,ctx_hn_bestrk,ctx_hn_margin,ctx_hn_soft05,ctx_hn_e1_rrw,ctx_hs_mutual,ctx_hs_k128,ctx_hs_e1,ctx_hs_t10` (variants abl_xdrop_ctxhn_hs_s23 / s24) | the existing xdrop mechanism: 24 xt columns left, xt_rel 5 (re-indexed), idcols 8, n_dense 104 -> 94; no new code. Completes the input ledger (pop / ctx_nd / ctx_hn+hs retrained-without on the current recipe) |

**Rules (the FROZEN abl_eval.py docstring rules of 00:08 UTC apply UNCHANGED; nothing is added or loosened):**
- retr_cat (SCALE-UP): PASS iff pooled val delta (LAST checkpoints, paired per feed vs abl_base_s23 / s24, pooled over
  seeds) > +1 SE_cal(val) AND pooled holdout delta >= 0 -> 10-epoch pair; info flags STRICT-PROMOTE and COLD/WARM TRADE as
  frozen. Prior (ledger §3): 0..+0.0008.
- xdrop_ctxhn_hs (ABLATION): LOAD-BEARING iff val delta < -2 SE_cal AND holdout delta < 0; DISPENSABLE iff |val delta| <= 1
  SE_cal AND holdout >= -1 SE_cal AND no val slice < -2 SE_feed; else INCONCLUSIVE. Worth = -delta. Diagnostics: stale-val
  (stale_pred.py, 'both_pre2both') + PRIVATE TRADE, pmix, levers, CLIPLOG, minutes ratio. Prior (replay reliance ctx_hn
  -0.0015 / ctx_hs -0.0032): -0.0005..-0.003.
- Readout file: logs/abl_eval_wave2.txt (+ .json) via `OUT=... abl_eval.py screen retr_cat xdrop_ctxhn_hs` (the wave-1
  logs/abl_eval.txt is not overwritten). Primary = LAST checkpoint; selected = info.
- f1..f5 REPLAY (f_replay.py; no training; replay-only hook on b['cf'] in a patched SplitData.gather, never imported by
  train.py): checkpoints abl_base_s23@last and abl_base_s24@last; modes ref (must reproduce the stored val / holdout
  preds), zero_all, perm_all (within-feed permutation of the f1..f5 rows among valid candidates, seed 2026+101k),
  zero_f1..f5, perm_f1..f5. Read: delta = mode - ref paired per feed on the same checkpoint, pooled over the two
  checkpoints, SE_feed (no run term), val slices + holdout all / late20%. RELIANCE READ ONLY: no gate, no recipe action
  (replay != retrain). Output logs/f_replay.txt (+ .json), preds under nn_x/abl/preds/replay/.
- GPUs: lanes on 5 / 7 (2 runs per GPU), replay on 3; nvidia-smi check (< 4 compute processes, >= 60 GB free) before every
  launch; PIDs in RUNNING.txt; nothing of other lanes touched (GPUs N carry the lead's 10-epoch pairs).

### 6.1 Added arm xdrop_pre2 (coordinator request; pre-declared 2026-11-13T04:18:51Z, BEFORE its first run)

| arm | type | exact change vs the base (A_BASE + HYG, seeds 23 / 24) | mechanism |
|---|---|---|---|
| xdrop_pre2 | ABLATION | append `--xt_exclude pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32,has:pre2_fe` (variants abl_xdrop_pre2_s23 / s24) | drops EVERY column of the cvnd_pre2s preset that comes from the pre-trained sequence encoder: the 6 pre2* columns (pre2_hmax32 is the encoder's hmax column; the preset holds no fe_noid / idmax* columns). The remaining 28 columns == the cvnd preset; XT_REL reduces to seq_cv_invd20 + it_hist_all_luo (2), XT_RELONLY empty, idcols 8, n_dense 104 -> 95. Existing --xt_exclude mechanism, no new code (test C1 layout check added). Purpose: the worth of the existing pre-trained encoder's scalar inputs on the current recipe (calibration for a separate pre-training study). Prior: pre2s vs cvnd +0.00191 ± 0.00058 at 3 ep (2 seeds, OLDER-ish), +0.0009 / +0.0005 at 10 ep -> expected 2-ep delta -0.001..-0.002 val |

Rule: the frozen ABLATION rule (LOAD-BEARING iff val delta < -2 SE_cal AND holdout delta < 0; DISPENSABLE iff |val delta| <= 1
SE_cal AND holdout >= -1 SE_cal AND no val slice < -2 SE_feed; else INCONCLUSIVE; worth = -delta), same readout (LAST
checkpoint primary, selected info, stale-val + PRIVATE TRADE as an input arm, pmix, levers, CLIPLOG, minutes) in
logs/abl_eval_wave2.txt together with retr_cat and xdrop_ctxhn_hs. Runs on GPU N (idle after the replay), 2 lanes, so the
running arms on GPUs N are not slowed.

### 6.2 SECOND-WAVE RESULTS (written 2026-11-13T05:07:40Z; read out 05:05:50 UTC by the frozen rules; logs/abl_eval_wave2.txt + .json, pre-stale copy
abl_eval_wave2_prestale.txt; replay logs/f_replay.txt + .json; freeze logs/FROZEN_abl_wave2.txt; tests CPU 96/96, GPU 20/20)

Runs (LAST checkpoint = step 19,670; selected = holdout-best at >= 12k): abl_retr_cat_s23 0.39444 / 0.31758, s24 0.39361 /
0.31750 (31 min, 2 per GPU on GPU N, 57.1 / 55.6 GB); abl_xdrop_ctxhn_hs_s23 0.39358 / 0.31778, s24 0.39252 / 0.31727 (24-25
min, GPU N); abl_xdrop_pre2_s23 0.39214 / 0.31743, s24 0.39243 / 0.31655 (24 min, GPU N). Base (wave 1): 0.39364 / 0.31754,
0.39348 / 0.31754. All 6 runs rc 0, selected step >= 19,000.

| arm | type | val delta (SE_cal, z_cal; seeds) | holdout delta (z_cal; seeds) | slices (val) | verdict |
|---|---|---|---|---|---|
| retr_cat | SCALE-UP | +0.00046 (0.00049, +0.93; +0.00080 / +0.00013) | +0.00000 (+0.01; +0.00003 / -0.00002), late20% -0.00034 | M<50 +0.00122 (z 1.5), M>=50 -0.00006, Tue +0.00125 (z 2.1), Thu -0.00040, coldpos +0.00030, warmpos +0.00045; pmix +0.00052 | **FAIL** (needs val > +1 SE_cal = +0.00049 AND holdout >= 0; val just below the bar, holdout flat); not STRICT-PROMOTE, no COLD/WARM TRADE. Selected read (info): +0.00054 / -0.00001. CLOSED by the frozen rule; recorded. Cost: ~1.25x step time at equal GPU load (31 min vs 24.5 min for the xdrop arms, both 2 per GPU; the 0.83 "ratio vs base" is confounded: the base ran 4 per GPU), memory +0 GB (57.1 vs 57.2 GB) |
| xdrop_ctxhn_hs | ABLATION | -0.00051 (0.00057, -0.90; -0.00006 / -0.00096) | -0.00002 (-0.05; +0.00024 / -0.00028), late20% -0.00041 | M<50 +0.00055, M>=50 -0.00124 (z -2.2), Tue +0.00032, Thu -0.00142 (z -1.9), coldpos -0.00058, warmpos -0.00069; pmix +0.00012; stale-val -0.00043 ± 0.00053, PRIVATE TRADE +0.00008 | **INCONCLUSIVE** (worth val +0.00051, holdout +0.00002; not DISPENSABLE because the M>=50 slice is < -2 SE_feed). The 10 ctx_hn / ctx_hs columns carry at most ~+0.0005 on the current recipe; no action |
| xdrop_pre2 | ABLATION | -0.00128 (0.00061, -2.09; -0.00150 / -0.00106) | -0.00054 (-1.59; -0.00010 / -0.00097), late20% -0.00040 | M<50 -0.00095, M>=50 -0.00151 (z -2.5), Tue -0.00119, Thu -0.00138, coldpos -0.00138, warmpos -0.00158 (z -2.1); pmix -0.00138 ± 0.00085; stale-val -0.00188 ± 0.00056 (z -3.3), PRIVATE TRADE -0.00060 (the pre2-free model loses MORE under the private-like coverage shift: it leans harder on the popularity inputs) | **LOAD-BEARING** (worth val +0.00128 ± 0.00061, holdout +0.00054 ± 0.00034). The pre-trained encoder's 6 scalar inputs are worth ~+0.0013 val / +0.0005 holdout at 2 epochs on the current recipe (consistent with the 3-ep +0.0019 and 10-ep +0.0009 / +0.0005 pairs), all val slices negative; selected read -0.00160 / -0.00051 |

Input ledger on the current recipe (2-ep retrain-without, worth val / holdout): popularity 8 +0.00234 / +0.00743, ctx_nd 10
+0.00135 / +0.00030, pre2 6 +0.00128 / +0.00054, ctx_hn+hs 10 +0.00051 / +0.00002 (n.s.). Retrieval scale-ups now 0 / 3
(retr32, retr32u, retr_cat): depth, dedup and same-category restriction all fail the screen; the base retr16+gate itself
is worth +0.00337 / +0.00294. Same-category item evidence adds nothing the dense category-affinity block + base retrieval
do not already carry (prior 0..+0.0008 in ledger §3: the realised +0.0005 is inside it but below the bar).

**f1..f5 REPLAY (abl_base_s23@last + s24@last; delta = mode - ref paired per feed, pooled; SE_feed):** ref reproduces the stored
val and holdout preds exactly (max |d| 0). zero_all -0.01766 ± 0.00116 val (M<50 -0.0160, M>=50 -0.0188, coldpos -0.0329,
warmpos -0.0118) / -0.01657 ± 0.00074 holdout; perm_all -0.01813 ± 0.00115 / -0.01824 ± 0.00074. Per column (zero / perm,
val; holdout): f4 -0.01376 / -0.00948 (coldpos -0.0314); -0.01432 / -0.00720 >> f5 -0.00503 / -0.00624; -0.00447 /
-0.00831 > f1 -0.00342 / -0.00424; -0.00401 / -0.00356 > f3 -0.00289 / -0.00169; -0.00384 / -0.00299 > f2 -0.00075 /
-0.00095; -0.00070 / -0.00075. The singles sum to ~0.026 (zero) > 0.018 (all): the five scores are partly redundant; f4 is
the dominant production score and the cold-positive channel (its zeroing costs cold positives -0.031 vs warm -0.007).
Reliance read only (replay != retrain); no action. The production-ranker confound (f1/f3/f5 slopes stronger on production
feeds) remains a question for a retrain-without, not for this replay.

---------------------------------------------------------------------------------------------------------------------

## 7. pop1 ARM pre-declaration (written 2026-11-13T06:20:42Z (`date -u`), BEFORE the first pop1 run and before the GPU harness tests; nn_x/codes/followup/RESULT.txt section (4) "POP1 DESIGN"; CURRENT_STATUS Log 06:03 "PRE-DECLARED pop1 arm")

Trigger: the pop-free model cannot tell a noisy high engagement rate from a reliable one. NOISY-HIGH (f5 >= val q90 and
exposure n = tte_it_imp + it_hist_all_luo < 5; 4.8% of val rows, 1,179 positives) act/exp is 0.922 for the 10-epoch pop-free
train-only members (4) vs 0.966 for the same-seed base; in this 2-epoch screen (f04c, @last pools) xdrop_pop 0.938 vs base
0.969 (-0.031 +- 0.002) and RELIABLE-HIGH 1.045 vs 1.005 (+0.040 +- 0.003). pop1 gives the pop-free inputs ONE capped
exposure count. The cap is meant to return the reliability information without bringing back the high-end popularity ordering.

| arm | type | exact change vs the base (A_BASE + HYG, seeds 23 / 24) | mechanism |
|---|---|---|---|
| pop1 | SCALE-UP; PRIMARY reference arm = xdrop_pop | append `--xt_exclude seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo --xt_add xp_imp_c50` (variants abl_pop1_s23 / s24; the --xt_exclude list is byte-identical to the xdrop_pop arm's) | xt_names = the 26 xdrop_pop columns + xp_imp_c50 appended LAST (27); xt_rel = the 3 pre2 columns (unchanged), xt_relonly the same 3, xt_idcols None (unchanged); n_dense 92 -> 93. attach_xt default branch (clip +-10; values in [0, 3.932], nothing clipped); not an ID-dropout column, no rank / z channel |

**Feature** xp_imp_c50 = log1p(min(tte_it_imp, 50)) float32 (nn_x/abl/pop1/build_xp_imp_c50.py). tte_it_imp = the item's TRAIN
impressions (src/features_tte.py): train rows on day d count train days < d only, val / test rows count all train days. No
labels and no own-day rows are used, so it is leak-free. New files data/feats/{train,val,test}/xp_imp_c50.npy (exclusive
create) are hard-linked into nn/feat_cache/{train,val,test}/ (same inode, nlink 2), the same way it_hist_all_luo is exposed.
No existing entry was touched.
Parity (nn_x/abl/pop1/out/xp_imp_c50_parity.json):

| | train | val | test |
|---|---|---|---|
| mean | 1.713 | 1.570 | 1.262 |
| share at the cap (>= 50) | 14.3% | 11.4% | 9.0% |
| share zero | 31.7% | 34.6% | 46.2% |
| median raw count | 4 | 3 | 1 |

PSI train-vs-val is 0.011 on the count bins {0,1,2,3-4,5-9,10-19,20-49,50} and 0.010 on 10 train-quantile bins. The rule is
< 0.5, so it PASSES. PSI train-vs-test is 0.102, val-vs-test 0.058, and train-without-day-0 vs val 0.057. Train day-0 rows
(day 20702) number 23,755,568 (9.6% of train) and are all 0 by construction. Without them, train has 24.4% zero and 15.8% at the cap.

**Code** (flag-off identical; nothing under nn/ code, nn_x/r6, nn_x/w5 edited):
- train.py `--xt_add` (default '') with helper abl_arms.xt_add_names. It is called once, behind `if a.xt_add:` in the
  no-init branch, right after `if a.xt_exclude` and right before the `if xt_names:` miss check. The --init refusal guard
  now reads `a.xt_exclude or a.xt_rel_extra or a.retr_cat or a.xt_add`. The helper refuses: no preset, empty or duplicated
  names, a name already present, seq_cv* / it_hist* names, and names in XT_REL / XT_RELONLY[preset].
- abl_lane.sh: arm pop1 added (write-new + mv).
- abl_eval.py: arm pop1 added (SCALE-UP, stale-val read), plus REFS = {pop1: xdrop_pop}. The pre-declared block below is
  ADDED. The frozen rules are unchanged. A regression re-read of the three wave-2 arms differed from logs/abl_eval_wave2.txt only in the SUMMARY header wording; that wording was restored, and a confirming re-read is reported in 7.1.
- tests/test_abl.py: AST proof now 3 imports / 4 add_argument / 5 guards; new block C5; G1 / G2 extended.
  CPU 110/110 PASS (logs/test_abl_cpu.log, 06:20 UTC). The GPU part runs on GPU N before the first pop1 launch (result in 7.1).

**Rule: PRIMARY = pop1 vs xdrop_pop** (the frozen SCALE-UP form, same machinery as every arm-vs-base read in abl_eval.py)
- Read the LAST checkpoints (step 19,670). The delta is pop1 - abl_xdrop_pop on the same seed, per feed, pooled over seeds
  23 / 24. SE_cal = sqrt(SE_feed^2 + sigma_run^2 / 2), with sigma_run 0.00035 (val) / 0.00015 (holdout).
- **PASS iff pooled val delta > +1 SE_cal(val) AND pooled holdout delta >= 0.** On PASS, and if the final pop-free FT read
  (nn_x/codes/followup/f04_popfree.py, once the n2cpv3 FT files land) confirms the noisy-high deficit, the 10-epoch pop-free
  family is rerun with the same flags (E_BASE + HYG + `--xt_exclude <8> --xt_add xp_imp_c50`, seeds 951-954) and goes
  through decide9 / decide6.
- **FAIL** -> the pop-free family stays as it is.
- STRICT-PROMOTE and COLD/WARM TRADE flags are reported as information.
- The selected-checkpoint read is information only.

**Information readouts (reported, never gates)**
- pop1 - abl_base: the remaining gap. xdrop_pop - base was -0.00234 val / -0.00743 holdout. It is printed as "INFO (not a
  gate): would-PASS / would-FAIL".
- Slices: val M<50, M>=50, Tue, Thu, coldpos, warmpos; holdout late20%.
- pmix, levers, CLIPLOG, minutes, per-seed deltas.
- NOISY-HIGH / RELIABLE-HIGH / MID-LOW / NO-DATA act/exp of the pop1 pool vs xdrop_pop vs base (nn_x/abl/pop1/pop1_actexp.py).
  It uses the f04c definitions; its plumbing check on base / xdrop_pop reproduces f04c exactly: 0.969 / 0.938, 1.005 / 1.045.
  Mechanism expectation: pop1's NOISY-HIGH moves from 0.938 toward 0.969, and its RELIABLE-HIGH from 1.045 toward 1.005.
- stale-val + PRIVATE TRADE: stale_pred.py 'both_pre2both' on abl_pop1_s23 / s24, selected and @last, in both blocks.
  Caveat: xp_imp_c50 has no private-like counterfactual. Under the stale scenario it keeps its val value and only the pre2
  overrides act, exactly as for xdrop_pop. So the pop1-vs-xdrop_pop stale delta does not test xp_imp_c50's own coverage
  shift. The test-split parity above (PSI 0.10) is the only read of that shift.

**Prior.** Popularity is worth +0.0023 val / +0.0074 holdout at 2 epochs. xp_imp_c50 has within-feed AUC 0.528 (popularity
0.547) and Spearman 0.744 with it_hist_all_luo. Expected pop1 - xdrop_pop: +0.0003..+0.0015 val; holdout positive and
probably larger (in-period exposure tracks popularity).

**Readout files:**
- logs/abl_eval_pop1.txt (+ .json; pre-stale copy abl_eval_pop1_prestale.txt), via `OUT=... abl_eval.py screen pop1`
- nn_x/abl/pop1/out/pop1_actexp.{log,json}
- waiter nn_x/abl/pop1/wait_pop1.sh (log nn_x/abl/pop1/wait_pop1.log)

**GPUs:**
- abl_lane.sh pop1:23 on GPU N and pop1:24 on GPU N, one per GPU. The lane checks nvidia-smi before each launch (< 4 compute
  processes, >= 60 GB free).
- CUDA_VISIBLE_DEVICES = one id; 12 CPU threads.
- stale_pred.py runs on one of GPUs N.
- PIDs are recorded in RUNNING.txt. Nothing on GPUs N is touched.

**7.0 Amendment (process only; written 2026-11-13T06:34:07Z, BEFORE the first pop1 launch; rule, arm and readouts unchanged).** The host CPU
load (other jobs, a high load average on a shared machine) slows data loading. The GPU harness test started 06:20:59 UTC on
GPU N (PID 3049408) has spent >15 min loading data for its FIRST end-to-end run (each run took ~1 min at 04:2x), so the GPU
part would need ~2 h. The two pop1 screen runs are therefore launched NOW, in parallel with the GPU tests, instead of after
them. If any GPU check fails, both pop1 runs are void: they are re-run after the fix, and nothing is read out from them.
The CPU suite (110/110, incl. the AST flag-off proof, the pop1 layout, the dense-feature and the checkpoint round-trip checks)
and the regression re-read are complete. The regression re-read of the wave-2 arms with the new abl_eval.py is IDENTICAL to
logs/abl_eval_wave2.txt (diff empty).

### 7.1 pop1 RESULTS (written 2026-11-13T07:08:32Z; read out 07:06:36 UTC by the pre-declared rule; logs/abl_eval_pop1.txt + .json, pre-stale copy abl_eval_pop1_prestale.txt; nn_x/abl/pop1/out/)

Tests: CPU 110/110 PASS (06:20); GPU (GPU N) 25/25 PASS (07:01). The GPU part includes flags-off bit-identity to frozen w5 (CD +
from scratch) and the arm-on run abl_t_fs_pop1 (cfg = the pop1 layout, 27 xt / n_dense 93; same batch order; stale_pred
from the cfg). The regression re-read of wave 2 with the new abl_eval.py is IDENTICAL. Runs: abl_pop1_s23 (GPU N, 16.4 min)
and abl_pop1_s24 (GPU N, 14.6 min), rc 0, selected = last = step 19,670: val 0.39278 / 0.39199, holdout 0.31154 / 0.31156.
For reference, xdrop_pop @last was 0.39141 / 0.39103 val and 0.30977 / 0.31044 holdout; base 0.39364 / 0.39348 and 0.31754 / 0.31754.

| comparison (LAST, pooled s23/s24) | val delta (SE_cal, z_cal; seeds) | holdout delta (z_cal; seeds) | slices (val; holdout late20%) | pmix | stale-val (PRIVATE TRADE) | verdict |
|---|---|---|---|---|---|---|
| **pop1 - xdrop_pop (PRIMARY)** | **+0.00116** (0.00065, +1.78; +0.00137 / +0.00095) | **+0.00144** (+3.79; +0.00176 / +0.00112) | M<50 +0.00075, M>=50 +0.00144 (z 2.2), Tue +0.00171 (z 2.0), Thu +0.00056, coldpos -0.00128 (z -1.1), warmpos +0.00236 (z 2.9); late20% +0.00091 | +0.00128 ± 0.00093 | +0.00112 ± 0.00060 (-0.00004) | **PASS** [strict-promote] [COLD/WARM TRADE flag: \|cold - warm\| = 0.0036 > 2 SE; the required stale-val and pmix reads are both positive] |
| pop1 - base (info) | -0.00118 (0.00092, -1.29; -0.00086 / -0.00150) | -0.00599 (-10.11) | coldpos +0.02144, warmpos -0.01046, Thu -0.00284 (z -2.2); late20% -0.00298 | +0.00013 ± 0.00132 | +0.00023 ± 0.00090 (+0.00141*) | INFO: would-FAIL. The gap to base shrinks from -0.00234 / -0.00743 (xdrop_pop) to -0.00118 / -0.00599 |

Selected-checkpoint read (info), pop1 - xdrop_pop: +0.00180 / +0.00138. Levers (pop1 - xdrop_pop): warm|recent +0.00104,
cat#1|w3+ +0.00076, warm|novel +0.00047, cold|* <= 0. *The stale-val and PRIVATE TRADE vs base are biased toward pop1:
the stale scenario degrades the base's popularity inputs to private-like coverage, but xp_imp_c50 has no counterfactual and
stays at its val value.

**Mechanism readout (pre-declared information, pop1_actexp.py, f04c definitions, @last 2-member pools):**

| cell | base | xdrop_pop | pop1 | pop1 - xdrop_pop |
|---|---|---|---|---|
| NOISY-HIGH act/exp | 0.969 | 0.938 | **0.938** | -0.000 ± 0.001 (share of the gap closed -0.01) |
| RELIABLE-HIGH act/exp | 1.005 | 1.045 | 1.041 | -0.004 ± 0.002 (0.10) |
| MID/LOW act/exp | 1.002 | 1.001 | 1.001 | 0.000 |
| NO-DATA act/exp | 0.930 | 0.976 | 0.988 | +0.013 ± 0.005 |

The expected mechanism did NOT materialise: noisy-high stays at 0.938. pop1's gain over xdrop_pop sits in warm positives
(warmpos +0.0024, warm|recent levers) and costs cold positives. That is a partial return of in-period popularity /
exposure ordering, not a reliability correction of f5.

**Final pop-free FT read (another agent's f04_popfree.py, 06:38 UTC; CURRENT_STATUS 06:39).** The FT pool's NOISY-HIGH
act/exp is 0.954 ± 0.025, so the pre-declared prediction (< 0.95) DOES NOT HOLD after fine-tuning. The pop1 PASS
consequence was conditioned on "the final FT read confirms the deficit", so that condition is NOT met. The 10-epoch rerun
(seeds 951-954, same flags) is therefore not triggered automatically; it is the lead's call.

**POST-HOC information (not pre-declared, no gate; pop1/pop1_diversity.py).** Does pop1 lose the pop-free family's
diversity? Barely. Rank-z pool correlation with base: val 0.9774 (xdrop_pop 0.9749), holdout 0.9693 (xdrop_pop 0.9652).
In a 2-family blend rz-mean(base pool, X), pop1 vs xdrop_pop gives val -0.00005 ± 0.00044 and holdout +0.00032 ± 0.00028.
The blend vs the base pool alone: val +0.00074 (pop1) / +0.00079 (xdrop_pop); holdout -0.00084 / -0.00115.
pop1 keeps the diversity and is stronger alone (+0.0012 val / +0.0014 holdout). Its gain is on the holdout axis, which is
the axis where the pop-free hedge Q1a failed (H -0.00052, z -2.45, CURRENT_STATUS 06:39).

### 7.2 Lead pop1 10-EPOCH FAMILY + stage-2 chain (lead decision PRE-DECLARED in CURRENT_STATUS 07:10 UTC; launched 07:11 UTC; recorded 2026-11-13T07:17:32Z)
- **Runs:** abl_e10_pop1_s951 on GPU N (PID 3906791), s952 on GPU N (3909444), s953 on GPU N (3910095), s954 on GPU N (3911266).
  They use the stage-E command of ABLATION_PLAN_1113 §3.1 (E_BASE + HYG, no --no_test) with the pop1 flags appended last
  (`--xt_exclude <8 POP> --xt_add xp_imp_c50`), launched with nn_x/abl/pop1/launch_e10.sh. Each GPU had 0 compute processes
  and [n] GB free before launch. Logged args differ from abl_e10_xdrop_pop_s95x ONLY in the variant and xt_add (plus the
  retr_cat key at its off value 0). 27 xt columns, 98,350 steps.
- **finetune.py extension (nn_x/abl/popfree/finetune.py, write-new + mv, 07:15 UTC).** honour_xt_exclude now also honours a
  checkpoint's --xt_add (same helper, same reserved set). CPU check nn_x/abl/pop1/chain/check_ft_honour.py: ALL PASS.
  - abl_pop1_s23 is accepted (27 xt, n_dense 93); the old code refused it.
  - abl_xdrop_pop_s23 / base_s23 / xrel_more_s23 / e10_xdrop_pop_s951: new == old (return value and log line).
  - Three tampered pop1 cfgs are refused. CUDA was never initialised.
  - GPU FT smoke on GPU N (abl_pop1_s23, nopop extra, 1 epoch, 2 folds; preds deleted, log logs/abl_pop1_s23_ftsmoke_ft.log):
    27 xt, n_dense 93, epoch-0 OOF 0.39278 = the stored train-only val NDCG.
- **Chain** nn_x/abl/pop1/chain/chain.sh with ARM=pop1 (PID 3970242; log pop1/chain/chain_pop1.log). It is a copy of the
  popfree chain generalised by ARM (default xdrop_pop). It waits for the members, runs readout.py, then does the FT
  (n2cpv3 + n2cpv3np, one member per GPU on 3/4/5/7, gpu_ok.sh before each launch), the copies + <v>np aliases into preds/
  (and new logs into nn/logs/ for presub_gate), and finally decide9.py pop1.
  - Candidates vs G3 (20 = F10 + F7x + F7x70):
    - Q2a = G3 + abl_e10_pop1_s951-954 (standard n2cpv3 FT).
    - Q2b = G3 + abl_e10_pop1_s951np-954np (nopop n2cpv3np FT alias members).
    - Q2c = G3 + abl_e10_xdrop_pop_s951-954 (standard FT, already in preds/) + abl_e10_pop1_s951-954 (standard FT).
  - Rule unchanged: H > +1 SE, V >= 0, no val slice < -2 SE, pmix >= -1 SE, then presub_gate on the pick only. Pick =
    highest H; tie -> higher pmix -> the nopop variant. Pick-1 replacement flag iff H >= 2 SE and V >= 0.
  - Tags Q2apop1 / Q2bpop1 / Q2cpop1 -> subs/cand_Q2a_G3pop1_memeq.parquet / cand_Q2b_G3pop1np_memeq.parquet /
    cand_Q2c_G3popfreepop1_memeq.parquet. A build is refused if the subs/ file or preds/ens_<TAG>_* already exist.
  - Checks:
    - ARM=xdrop_pop reproduces popfree/decide9_pick.json exactly (candidates identical).
    - A plumbing run with stand-ins (PF = the pop-free members, POPFREE = r6_cd_c0_r101/102) reproduces Q1a = 0.40068 / 0.32327.
    - Tie-break unit cases pass.
    - DRY runs with FAKE_PRESENT {all 4, 2, 1} pass.
  - Nothing is submitted.

## 8. pop2 ARM pre-declaration (written 2026-11-13T14:30:44Z (`date -u`), BEFORE the first pop2 run; POPULARITY-VIEW FAMILIES follow-up, CURRENT_STATUS Log 14:17 UTC, Task B; agent)

State at writing: no abl_pop2_* / abl_e10_pop2_* file exists in nn_x/abl (0 matches); GPUs N idle (0 compute processes, [n] GB
free); GPUs N run Task A (abl_e10_pop1_s955-958); GPUs N belong to the JOINT agent and are never used.

Idea: a SECOND popularity-view member family = the pop-free inputs (xdrop_pop: the 8 popularity / co-visitation columns removed)
plus ONE different train-period statistic. pop1 gave an exposure COUNT (xp_imp_c50); pop2 gives a CLICK RATE. On val the two
are a different view (Spearman xp_ctr_c vs xp_imp_c50 -0.265, vs it_hist_all_luo -0.124; xp_imp_c50 vs it_hist_all_luo +0.744).

| arm | type | exact change vs the base (A_BASE + HYG, seeds 23 / 24) | mechanism |
|---|---|---|---|
| pop2 | SCALE-UP; PRIMARY reference arm = xdrop_pop | append `--xt_exclude seq_cv_invd20,seq_cv_pop,seq_cv_norm,it_hist_all_luo,it_hist_et1_luo,it_hist_et2_luo,it_hist_et3_luo,it_hist_et4_luo --xt_add xp_ctr_c` (variants abl_pop2_s23 / s24; the --xt_exclude list is byte-identical to xdrop_pop's and pop1's; abl_lane.sh DRY: the command equals pop1's except the variant and the --xt_add value) | xt_names = the 26 xdrop_pop columns + xp_ctr_c appended LAST (27); xt_rel / xt_relonly / xt_idcols unchanged; n_dense 92 -> 93; attach_xt default branch (clip +-10; values in [-2, 2], nothing clipped); no rank / z channel |

**Feature** xp_ctr_c = clip(log(tte_it_ctr / 0.0159), -2, +2) float32 (nn_x/abl/pop2/build_xp_ctr.py; PRIOR 0.0159 = src/features_tte.py).
tte_it_ctr = (it_pos + 0.0159 * 20) / (it_imp + 20) from TRAIN labels: train rows on day d use train days < d only (day-0 rows and
items without earlier impressions sit at exactly the prior -> xp_ctr_c = 0), val / test rows use all train days. No own-row / own-day
labels, no later snapshots -> leak-free. +-2 = a click rate 7.4x above / below the prior (ctr 0.117 / 0.00215), reached only by
>= 3-4 all-clicked or >= 128 never-clicked impressions, so the clip removes the extreme popularity ordering at both ends.
New files data/feats/{train,val,test}/xp_ctr_c.npy (exclusive create, 14:26 UTC) hard-linked into nn/feat_cache/{train,val,test}/
(same inode, nlink 2), as xp_imp_c50. No existing entry was touched.
Parity (nn_x/abl/pop2/out/xp_ctr_c_parity.json; transform exploration in the session scratchpad explore_ctr.py, 14:25 UTC):

| | train | val | test |
|---|---|---|---|
| mean / sd | -0.017 / 0.566 | -0.054 / 0.515 | -0.050 / 0.456 |
| share at -2 (clip) / at +2 (clip) | 0.34% / 0.04% | 0.30% / 0.01% | 0.25% / 0.01% |
| share at the prior (no earlier train impressions) | 31.7% | 34.6% | 46.2% |

PSI train-vs-val 0.014 (13 natural log-ratio bins incl. the clip atoms and the prior atom) / 0.014 (10 train-quantile bins): PASSES
the project rule (< 0.5) and matches xp_imp_c50's 0.011. PSI train-vs-test 0.096 / 0.105 = xp_imp_c50's 0.102; it is driven entirely
by the no-train-impression atom (31.7% -> 46.2%), which no transform of a train-period statistic can change (every clip of the log
ratio, the uncapped ratio and linear caps all give 0.014 / 0.105). DISCLOSED: the follow-up brief asked for train-vs-test < 0.05
'like xp_imp_c50'; xp_imp_c50 itself is at 0.102, so the brief's figure was its train-vs-val PSI, and pop2 is at parity with pop1
on both reads. Transform choice: all clips tie on PSI; +-2 is the widest symmetric clip that still trims the extreme tails
(train quantiles p0.1 -2.51 / p99.9 +1.89; 99.6% of the rows unclipped). Train day-0 rows (9.6%) are all at the prior.
Information: within-feed val AUC of the raw ctr 0.510 (xp_imp_c50 0.527, popularity 0.547): a weak cue alone.

**Code** (train.py / abl_arms.py / popfree/finetune.py untouched: --xt_add is generic and honours any feat_cache column):
- abl_lane.sh: arm pop2 appended (write-new + mv at 14:28:46Z, md5 9c6402e283760cfb15ac9b37eaa2d8c2); DRY verified as above;
  bash -n OK; a GPU outside ABL_GPUS is refused.
- abl_eval.py: ARMS pop2 (SCALE-UP, stale-val read), REFS = {pop1: xdrop_pop, pop2: xdrop_pop}, REF_PASS pop2, docstring block
  POP2 ARM (write-new + mv, md5 c08d1daeb95fb06382755922f068892f; import check OK). The frozen rules are unchanged.
- tests/test_abl.py cpu is re-run in the background (logs/test_abl_cpu_pop2.log); its result is recorded in 8.1. No GPU test
  (no code under test changed; the pop1 GPU tests of 7.1 cover the --xt_add path).

**Rule: PRIMARY = pop2 vs xdrop_pop** (the frozen SCALE-UP form of abl_eval.py, exactly as section 7 for pop1)
- Read the LAST checkpoints (step 19,670). The delta is pop2 - abl_xdrop_pop on the same seed, per feed, pooled over seeds 23 / 24.
  SE_cal = sqrt(SE_feed^2 + sigma_run^2 / 2), sigma_run 0.00035 (val) / 0.00015 (holdout).
- **PASS iff pooled val delta > +1 SE_cal(val) AND pooled holdout delta >= 0.** If the COLD/WARM TRADE flag is raised, the stale-val
  and pmix reads are required (as 7.1 applied them): a PASS stands only if neither pop2-vs-xdrop_pop stale-val nor pmix is < -1 SE.
- **FAIL** -> Task B stops; nothing else is run.
- STRICT-PROMOTE flag, the selected-checkpoint read, pop2 - abl_base ('would-PASS / would-FAIL'), slices, pmix, stale-val, PRIVATE
  TRADE (caveat as for xp_imp_c50: xp_ctr_c has no private-like counterfactual, so the stale scenario tests only the pre2 overrides),
  levers, CLIPLOG, minutes, per-seed deltas are information.
- Readout files: logs/abl_eval_pop2.txt (+ .json; pre-stale copy abl_eval_pop2_prestale.txt) via `OUT=... abl_eval.py screen pop2`
  from the waiter nn_x/abl/pop2/wait_pop2.sh (log nn_x/abl/pop2/wait_pop2.log; stale_pred.py on GPU N or 7).

**On PASS (pre-declared consequence):** the 10-epoch family abl_e10_pop2_s951-954 = E_BASE + HYG + the pop2 flags appended last
(nn_x/abl/pop2/launch_e10_pop2.sh = nn_x/abl/pop1/launch_e10.sh with --xt_add xp_ctr_c; args must equal abl_e10_pop1_s951's except
variant / seed / xt_add), one run per GPU on 6 / 7 and on 0 / 1 / 4 / 5 once Task A's runs and FTs have finished (<= 3 processes,
>= 60 GB free before each launch); standard n2cpv3 FT only (x8/ft_list_x8.sh); copies cp --update=none into preds/, preds/holdout/,
nn/logs/; ONE read-out nn_x/abl/pop1/x8/decide_x8.py Q3 (pools exactly as decide9.py): Q3 = G3 + pop2(951-954) [24], Q3b = G3 +
pop1(951-954) + pop2(951-954) [28]. Each QUALIFIES iff (a) V - V(G3) > +1 SE; (b) V - V(G3) >= -1 SE; (c) pmix(X - G3) >= -1 SE;
(d) no val slice (M<50, M>=50, Tue, Thu) < -2 SE; (e) in-period holdout X - G3 >= -1 SE (a veto: the holdout lies inside the train
period, contains only M>=50 feeds and under-values popularity-light members); (f) presub_gate PASS (the pop1 chain's exact command,
refs ens_G3_*). A qualifier is built with popfree/build_pool.py (TAG Q3pop2 -> subs/cand_Q3_G3pop2_memeq.parquet; TAG Q3bpop12 ->
subs/cand_Q3b_G3pop1pop2_memeq.parquet; refused if the files exist) and gated; it is a recorded candidate, never a pick-1
replacement by itself; nothing is submitted. Minimum 2 complete pop2 members (disclosed if fewer than 4).

**Prior.** pop1 passed the same screen with +0.0012 val / +0.0014 holdout vs xdrop_pop; the click rate is a weaker cue (AUC 0.510 vs
0.527) but a different one. P(screen PASS) ~0.35. The 10-epoch family will be weaker than the base (pop1: -0.0024 val / -0.0062
holdout per member); P(Q3 or Q3b qualifies) ~0.15; expected private effect of a qualifier ~ +0.0002 +- 0.0003.

**GPUs:** abl_lane.sh pop2:23 on GPU N and pop2:24 on GPU N (ABL_GPUS="6 7"; the lane checks <= 3 compute processes and >= 60 GB
before the launch); CUDA_VISIBLE_DEVICES = one id; 12 CPU threads; PIDs in RUNNING.txt (train.py PIDs verified via /proc); stop
by exact PID only. Nothing on GPUs N is touched.

### 8.1 pop2 SCREEN RESULTS (written 2026-11-13T14:47:31Z; read out 14:45:40 UTC by the pre-declared rule; logs/abl_eval_pop2.txt + .json, pre-stale copy abl_eval_pop2_prestale.txt)

Tests: tests/test_abl.py cpu re-run with the pop2 arm in the lane: 110/110 PASS (logs/test_abl_cpu_pop2.log, 52 s). Runs: abl_pop2_s23
(GPU N, 12.2 min) and abl_pop2_s24 (GPU N, 12.1 min), rc 0, launched 14:30:45 UTC, selected = last = step 19,670: val 0.39308 / 0.39254,
holdout 0.31168 / 0.31119 (xdrop_pop @last 0.39141 / 0.39103 and 0.30977 / 0.31044; base 0.39364 / 0.39348 and 0.31754 / 0.31753;
pop1 @last 0.39278 / 0.39199 and 0.31154 / 0.31156). Both runs logged 'abl xt_add: appended 1 xt columns ['xp_ctr_c'] after the 26
kept' (27 xt, n_dense 93). stale_pred.py on GPU N (PID 1116678, 9 s).

| comparison (LAST, pooled s23/s24) | val delta (SE_cal, z_cal; seeds) | holdout delta (z_cal; seeds) | slices (val; holdout late20%) | pmix | stale-val (PRIVATE TRADE) | verdict |
|---|---|---|---|---|---|---|
| **pop2 - xdrop_pop (PRIMARY)** | **+0.00159** (0.00063, +2.52; +0.00167 / +0.00151) | **+0.00133** (+3.66; +0.00191 / +0.00075) | M<50 +0.00092, M>=50 +0.00205 (z 3.2), Tue +0.00201 (z 2.5), Thu +0.00113, coldpos -0.00125 (z -1.1), warmpos +0.00286 (z 3.6); late20% +0.00110 | +0.00242 +- 0.00092 (z 2.7) | +0.00147 +- 0.00058 (z 2.6) (-0.00012) | **PASS** [strict-promote] [COLD/WARM TRADE flag: the required stale-val and pmix reads are both positive] |
| pop2 - base (info) | -0.00075 (0.00094, -0.80; -0.00056 / -0.00094) | -0.00610 (-10.14; -0.00586 / -0.00634) | coldpos +0.02148, warmpos -0.00996, Thu -0.00228 (z -1.8); late20% -0.00279 | +0.00127 +- 0.00135 | +0.00058 +- 0.00092 (+0.00133*) | INFO: would-FAIL. Gap to base -0.00075 / -0.00610 (pop1: -0.00118 / -0.00599; xdrop_pop: -0.00234 / -0.00743) |

Selected-checkpoint read (info), pop2 - xdrop_pop: +0.00223 / +0.00126. Levers (pop2 - xdrop_pop): warm|recent +0.00137, cat#1|w3+
+0.00061, warm|novel +0.00044, cold|recent -0.00033. *The stale-val vs base is biased toward pop2 as for pop1 (xp_ctr_c has no
private-like counterfactual). Reading: the click rate returns the same warm-positive ordering as the exposure count (warmpos +0.0029
vs +0.0024 for pop1) and is a slightly stronger member at 2 epochs (val gap to base -0.0008 vs -0.0012).

**Consequence (pre-declared):** the 10-epoch family abl_e10_pop2_s951-954 is launched: s951 on GPU N and s952 on GPU N at ~14:47 UTC
(pop2/launch_e10_pop2.sh; args checked = abl_e10_pop1_s951's except variant / seed / xt_add); s953 (GPU N) and s954 (GPU N) are queued
behind Task A's FT by pop2/queue_s953_s954.sh; stage-2 chain pop2/chain_pop2.sh (FT n2cpv3 only, copies, ONE read-out
pop1/x8/decide_x8.py Q3 = G3 + pop2(4) and Q3b = G3 + pop1(4) + pop2(4) under the section-8 rule, build + presub_gate on qualifiers).
decide_x8.py was plumbing-tested with the pop-free members standing in for the second family: it reproduces decide9's Q2a / Q1a / Q2c
numbers and family correlations exactly (x8/decide_x8_Q2a8_plumbing.json, x8/decide_x8_Q3_plumbing.json).

### 8.2 Process amendment (written 2026-11-13T14:51:11Z; GPUs only, rule / arms / readouts unchanged)
The coordinator freed GPUs N (the JOINT agent finished). abl_e10_pop2_s953 and s954, pre-declared for GPUs N after Task A's FT,
were launched instead on GPU N and GPU N at ~14:51 UTC under the same rules (0 compute processes, [n] GB free before each launch;
one CUDA_VISIBLE_DEVICES id; PIDs verified via /proc and recorded in RUNNING.txt); the queue pop2/queue_s953_s954.sh was stopped by
exact PID (1118658) before it could launch anything. pop2/launch_e10_pop2.sh: the allowed-GPU list extended to 0-7 (write-new + mv).

### 8.3 pop2 10-EPOCH FAMILY RESULTS (written 2026-11-13T15:59:28Z; single read-out 15:57:13-15:57:32Z by pop2/chain_pop2.sh, rule section 8; pop1/x8/decide_x8_Q3.json, pop2/chain_pop2.log)

Runs abl_e10_pop2_s951 (GPU N) / s952 (GPU N) launched 14:46:56Z, s953 (GPU N) / s954 (GPU N) 14:50:43Z (8.2); all complete 15:54:03Z
(~63-67 min each; args = abl_e10_pop1_s951's except variant / seed / xt_add). FT n2cpv3 one member per GPU on 6/7/0/1, 15:54:23-15:57:13Z.

| member | train-only val | holdout | FT OOF (final EMA; best epoch) |
|---|---|---|---|
| abl_e10_pop2_s951 | 0.39429 | 0.31299 | 0.39539 (0.39554 ep1) |
| abl_e10_pop2_s952 | 0.39395 | 0.31414 | 0.39538 (0.39544 ep2) |
| abl_e10_pop2_s953 | 0.39509 | 0.31309 | 0.39565 (0.39606 ep1) |
| abl_e10_pop2_s954 | 0.39556 | 0.31348 | 0.39615 (0.39680 ep1) |
(pop1 s951-954: to val 0.39214-0.39581, holdout 0.31375-0.31437, FT OOF 0.39345-0.39686; base w5_e10 to val 0.39549-0.39745, holdout 0.32000-0.32084)

| pool (member-equal 0.5 z(ft) + 0.5 z(to); paired vs G3 0.40032 / 0.32379) | n | val | hold | V | H | pmix | slices z (M<50, M>=50, Tue, Thu) | corr val / hold | rule |
|---|---|---|---|---|---|---|---|---|---|
| Q3 = G3 + pop2(951-954) | 24 | 0.40047 | 0.32356 | +0.00015 +- 0.00031 (z 0.5) | -0.00023 +- 0.00022 (z -1.07) | +0.00084 +- 0.00049 | -1.7, +2.5, -0.1, +0.8 | 0.9995 / 0.9991 | (a) FALSE, (e) FALSE -> NOT a candidate |
| Q3b = G3 + pop1(4) + pop2(4) | 28 | 0.40022 | 0.32317 | -0.00010 +- 0.00040 | -0.00062 +- 0.00029 (z -2.14) | +0.00052 +- 0.00061 | -2.1, +2.1, -0.3, -0.1 | 0.9987 / 0.9977 | (a), (d), (e) FALSE -> NOT a candidate |

Nothing built (no subs/cand_Q3_* / cand_Q3b_*, no preds/ens_Q3pop2_* / ens_Q3bpop12_*), nothing submitted. Family pool z-corr to the G3
train-only pool: pop2 0.9754 (val) / 0.9677 (holdout) (pop1 0.9760 / 0.9691); pop2 pool vs pop1 pool 0.9863 / 0.9858 = the pop1
seed-half correlation (pop1(955-958) vs pop1(951-954) 0.9864 / 0.9863). READING: the click-rate input and the exposure-count input
produce the SAME member view (a weaker-than-base near copy of G3, corr 0.9995), so pop2 adds nothing that pop1 did not; the M>=50 /
pmix gains (+0.0009 / +0.0008) are the known long-feed pattern of popularity-light members and are offset on the in-period holdout.
Together with Task A (x8/RESULT_pop1x8.txt: Q2a8 vetoed by the holdout at z -2.5) the popularity-view family route is CLOSED:
pop-free (Q1a), pop1 x4 (Q2a), pop1 x8 (Q2a8), pop2 (Q3), pop1+pop2 (Q3b) are all 0 +- 0.0003 on val and <= 0 on the holdout.

## 9. TIME ARM pre-declaration (written 2026-11-13T19:16:34Z (`date -u`), BEFORE the first time run; CURRENT_STATUS Log 19:08 UTC "TIME BLIND SPOT FOUND"; agent)

State at writing: no abl_time_s* / abl_e10_time_* / Q4 file exists in nn_x/abl, preds/, subs/, nn/logs (find: 0 matches). The three
feature columns were BUILT at 19:12 UTC (a data build, not a run). GPUs N: another agent's small fine-tune jobs are present
(27.7 GB used on each; the lane's own check applies before each launch). Harness CPU tests re-run in the background with the new arm
(logs/test_abl_cpu_time.log; result recorded in 9.1).

Facts (lead, 18:50 UTC, CPU): the ranker has NO time-of-day / weekday input; train = Sep 6-19 2026 MSK, 31.5% weekend feeds; val =
Tue / Thu only (7,542 / 6,878); PUBLIC = Sat 6,050 + Sun 2,512; PRIVATE = Mon 7,578 / Tue 7,600 / Sun 4,855; category positive rates
differ weekend vs weekday (log-ratio sd 0.046 over 52 categories, range -0.12..+0.14, 10 at |z| > 3; within-feed spread median 0.08,
p90 0.17); the hour mix is similar across splits. The weekend x category interaction is INVISIBLE on val, so the pre-declared primary
read is the HOLDOUT WEEKEND SLICE (val = non-regression check); a positive arm becomes a PUBLIC-PROBE family (public is 100% weekend).

| arm | type | exact change vs the base (A_BASE + HYG, seeds 23 / 24) | mechanism |
|---|---|---|---|
| time | SCALE-UP; reference = base (the default comparison) | append `--xt_add tm_weekend,tm_hsin,tm_hcos` (variants abl_time_s23 / s24; nothing excluded; abl_lane.sh DRY: the command equals abl_base_s23's plus the --xt_add) | xt_names = the 34 cvnd_pre2s columns + the 3 time columns appended LAST (37); xt_rel / xt_relonly / xt_idcols unchanged; n_dense 95 -> 98; attach_xt default branch (clip +-10; values in {0,1} / [-1,1], nothing clipped); no rank / z channel; the dense MLP / cross blocks can form category x weekend / x hour interactions |

**Features** (nn_x/abl/time/build_tm.py, 19:12 UTC; parity nn_x/abl/time/out/tm_parity.json): per candidate, broadcast from the FEED's
timestamp (cand_fidx), Moscow local time = UTC + 3 h: tm_weekend = 1[weekday in Sat/Sun]; tm_hsin / tm_hcos = sin / cos(2 pi *
seconds-of-day / 86400). No labels / history / item statistics -> leak-free by construction. New files data/feats/{train,val,test}/
tm_{weekend,hsin,hcos}.npy (exclusive create) hard-linked into nn/feat_cache/{train,val,test}/ (same inode, nlink 2); nothing existing
touched. (data/feats/*/hour.npy, an integer Moscow hour, exists and is unused by every preset; src/presub_gate.py WARNs on the name
`hour` as drift-prone; the tm_* columns are a different name and a bounded encoding, and their drift is documented here.)

| | train | val | test | holdout (30K) |
|---|---|---|---|---|
| feeds per weekday | Mon 526K Tue 291K Wed 286K Thu 552K Fri 504K Sat 437K Sun 555K | Tue 7,542 Thu 6,878 (+3 midnight-edge rows) | Sat 6,050 Sun 7,367 Mon 7,578 Tue 7,600 | Mon 5,136 Tue 2,659 Wed 2,440 Thu 5,268 Fri 4,865 Sat 4,459 Sun 5,173 |
| weekend share feeds / rows | 0.3146 / 0.3214 | 0.0000 / 0.0000 | 0.4692 / 0.4788 | 0.3211 (9,632 weekend / 20,368 weekday feeds) |
| mean tm_hsin / tm_hcos (rows) | -0.174 / -0.095 | -0.162 / -0.188 | -0.185 / -0.058 | - |
| hour mix by 3-h block (feeds) | .052 .045 .129 .155 .160 .156 .154 .147 | .052 .040 .133 .178 .177 .170 .127 .123 | .055 .045 .124 .155 .138 .170 .149 .164 | .061 .045 .128 .145 .149 .150 .153 .169 |

PSI train-vs-val: tm_weekend 4.20 (rows) = BY CONSTRUCTION (val has 0 weekend feeds; a two-atom PSI with one empty atom) -> DOCUMENTED,
not a parity failure, and exactly why val cannot read the weekend interaction and the holdout weekend slice is the primary read;
hour blocks 0.016 (feeds) / 0.018 (rows). PSI train-vs-test: tm_weekend 0.104 (0.32 -> 0.48 weekend rows), hour blocks 0.007 / 0.006.
PSI val-vs-test hour blocks 0.028. The hour inputs pass the project rule (< 0.5) on every pair.

**Code** (train.py / abl_arms.py / popfree/finetune.py untouched: --xt_add takes a comma list, abl_arms.xt_add_names appends all names):
- abl_lane.sh: arm time appended (write-new + mv at 19:16 UTC, md5 1c933012ecd51853a8469e09af0e3af5); DRY verified (command = base +
  --xt_add tm_weekend,tm_hsin,tm_hcos); bash -n OK; the GPU allow-list is passed as ABL_GPUS="6 7".
- abl_eval.py: ARMS time (SCALE-UP, stale-val read as information), docstring block TIME ARM; REFS unchanged (reference = base = the
  default comparison) (write-new + mv, md5 c79d9580a416678ee86f05639d327232; import check OK). The frozen rules are unchanged.
- nn_x/abl/time/abl_eval_time.py: wrapper of abl_eval.screen(['time']) that adds the holdout slices weekend / weekday (R.setup('hold')
  wrapped; 9,632 / 20,368 feeds, asserted) and prints the pre-declared verdict below (smoke-tested on the incomplete state).
- nn_x/abl/time/wait_time.sh: copy of pop2/wait_pop2.sh (read-out, stale_pred on GPU N, read-out again).

**Rule: PRIMARY = time vs base, holdout WEEKEND slice** (LAST checkpoints, step 19,670; delta = abl_time_s<seed>@last - abl_base_s<seed>@last
per feed, pooled over seeds 23 / 24; slice SE = SE_feed = sd / sqrt(n_slice), the frozen slice convention; SE_cal = sqrt(SE_feed^2 +
sigma_run^2 / 2), sigma_run 0.00035 val / 0.00015 holdout):
- **PASS iff holdout WEEKEND-slice pooled delta > +1 SE_feed(weekend) AND holdout overall pooled delta >= 0 AND val pooled delta >= -1
  SE_cal(val) AND pmix pooled delta >= -1 SE(pmix).** (pmix unavailable -> no PASS.)
- **FAIL** -> the TIME arm stops; nothing else is run (the features stay on disk for any later use).
- Information (never gates): the holdout weekday slice and late20%, per-seed weekend / weekday deltas, the selected-checkpoint read, val
  slices (M<50, M>=50, Tue, Thu, coldpos, warmpos), stale-val if the waiter produces it (the stale scenario overrides popularity /
  pre2 inputs only; the tm_* inputs have no private-like counterfactual), levers, CLIPLOG, minutes, the standard SCALE-UP verdict.
- Readout files: logs/abl_eval_time.txt (+ .json; pre-stale copy abl_eval_time_prestale.txt) via nn_x/abl/time/wait_time.sh (log
  nn_x/abl/time/wait_time.log).

**On PASS (pre-declared consequence):** the 10-epoch family abl_e10_time_s951-954 = E_BASE + HYG + `--xt_add tm_weekend,tm_hsin,tm_hcos`
appended last (nn_x/abl/time/launch_e10_time.sh = pop2/launch_e10_pop2.sh without --xt_exclude and with the time --xt_add; the args must
equal abl_e10_pop2_s951's except variant / seed / the arm flags), on GPUs N ONLY: s951 + s953 on GPU N, s952 + s954 on GPU N, two
runs per GPU concurrently (each launch subject to <= 3 compute processes and >= 60 GB free, polled every 120 s up to 2 h; concurrency
is never slower in wall time than two sequential rounds and the 5-h cap is the binding constraint); standard n2cpv3 FT only
(pop1/x8/ft_list_x8.sh with ALLOWED_GPUS="6 7"; popfree/finetune.py honours the checkpoint's --xt_add: honour_xt_exclude recomputes
xt_names with abl_arms.xt_add_names and attach_xt loads them, verified by reading the code; the FT log line 'abl xt_exclude honoured'
must show 37 xt columns); copies cp --update=none into preds/, preds/holdout/, nn/logs/; ONE read-out nn_x/abl/time/decide_time.py Q4
(decide_x8.py's pool machinery verbatim + the holdout weekend / weekday slices): Q4 = G3 + time(951-954) [24 members, member-equal
0.5 z(ft) + 0.5 z(to)] vs G3 (20). Q4 QUALIFIES iff (a) in-period holdout WEEKEND-slice H_we(Q4 - G3) > +1 SE (primary); (b) holdout
overall H(Q4 - G3) >= -1 SE; (c) pmix(Q4 - G3) >= -1 SE (veto); (d) V(Q4 - G3) >= -2 SE and no val slice (M<50, M>=50, Tue, Thu) < -2 SE
(sanity veto added by the agent; V is expected ~0 and is otherwise information); (e) presub_gate PASS (the pop2 chain's exact command,
refs ens_G3_*). A qualifier is built with popfree/build_pool.py (TAG Q4time -> subs/cand_Q4_G3time_memeq.parquet; refused if the file
or preds/ens_Q4time_* exist) and gated (nn_x/abl/time/gate_Q4time.txt + sha256) as a PUBLIC PROBE for the user to submit; it is never a
pick-1 replacement by itself; nothing is submitted by the agent. Minimum 2 complete members (disclosed if fewer than 4). If the 5-h cap
(00:08 UTC) would be exceeded, the family proceeds with the complete members and the rest is reported as not run.

**Prior.** The CURRENT_STATUS external-data close (every data-derived calendar x category / x hour prior within +-0.0003, |z| < 2, on val
AND on the weekend-containing holdout) bounds the post-hoc version of this lever; the model-internal version can use interactions the
priors could not, but the within-feed log-rate spread (median 0.08) keeps it small. Weekend-slice SE_feed ~0.0006 at 9.6K feeds, so a
PASS needs > ~+0.0006 there; a null arm passes the one-sided +1 SE clause by chance 16% of the time. P(screen PASS) ~0.25 (incl. the
noise pass); P(Q4 qualifies) ~0.1; expected public effect of a qualifier +0.0002 +- 0.0005 (val-invisible by construction).

**GPUs:** abl_lane.sh time:23 on GPU N and time:24 on GPU N (ABL_GPUS="6 7"; the lane checks <= 3 compute processes and >= 60 GB before
the launch, polling every 120 s); CUDA_VISIBLE_DEVICES = one id; 12 CPU threads; setsid nohup; train.py PIDs verified via
/proc/<pid>/cmdline in RUNNING.txt; stop by exact PID only; nothing of any other process is touched. TIME CAP: 5 h from 19:08 UTC.

### 9.1 TIME ARM SCREEN RESULTS (written 2026-11-13T19:41:17Z; read out 19:39:17 UTC (pre-stale) and 19:40:35 UTC (final) by the pre-declared rule; logs/abl_eval_time.txt + .json, pre-stale copy abl_eval_time_prestale.txt; waiter log nn_x/abl/time/wait_time.log)

Tests: tests/test_abl.py cpu re-run with the time arm in the lane: 110/110 PASS (logs/test_abl_cpu_time.log, 46 s). Runs: abl_time_s23 (GPU N,
20.4 min) and abl_time_s24 (GPU N, 20.2 min), rc 0, launched 19:16:52 UTC via abl_lane.sh (3 other compute processes on each GPU at launch,
[n] GB free), both logged 'abl xt_add: appended 3 xt columns ['tm_weekend', 'tm_hsin', 'tm_hcos'] after the 34 kept (indices [34, 35, 36])'.
Selected = last (step 19,670) for s23; s24 selected step 19000. @last: val 0.39342 / 0.39404, holdout 0.31760 / 0.31748 (base @last 0.39364 /
0.39348 and 0.31754 / 0.31753). stale_pred.py on GPU N (PID 1975689, 45 s). (The base runs took 37 min under a heavier GPU load on 11-13
01:00; the 0.54 minutes ratio is load, not cost.)

| comparison (LAST, pooled s23 / s24; delta = time - base) | value | rule clause |
|---|---|---|
| **holdout WEEKEND slice (n 9,632)** | **+0.00021 +- 0.00051 (z +0.40**; z with SE_cal +0.39); per seed **+0.00144 / -0.00103** | > +1 SE_feed: **FALSE** (PRIMARY) |
| holdout weekday slice (n 20,368) | -0.00009 +- 0.00034 (z -0.26); per seed -0.00059 / +0.00041 | information |
| holdout overall | +0.00001 +- 0.00028 (z_cal +0.02); per seed +0.00006 / -0.00005; late20% -0.00028 (z -0.47) | >= 0: TRUE |
| val overall | +0.00017 (SE_cal 0.00055, z_cal +0.30); per seed -0.00022 / +0.00055 | >= -1 SE_cal: TRUE |
| pmix | -0.00056 +- 0.00075 (z -0.75) | >= -1 SE: TRUE |
| val slices | M<50 +0.00081 (z +0.9), M>=50 -0.00028 (z -0.5), Tue +0.00101 (z +1.5), Thu -0.00076 (z -1.1), coldpos -0.00101 (z -1.0), warmpos +0.00072 (z +1.1) | information |
| selected-checkpoint read (info) | val -0.00005 (z -0.10), holdout +0.00001, weekend +0.00026 +- 0.00053 (z +0.49; per seed +0.00137 / -0.00086), weekday -0.00011 | information |
| stale-val (info; the stale scenario overrides popularity / pre2 inputs only) | -0.00080 +- 0.00050 (z -1.60; selected -0.00059, z -1.15); PRIVATE TRADE -0.00096 | information |
| levers (info) | warm\|recent +0.00045 +- 0.00039, cold\|recent -0.00037 +- 0.00025, total +0.00013 +- 0.00049 | information |
| CLIPLOG | plateau clipped 0.000 on all four runs | closed |

**VERDICT (pre-declared rule): FAIL** -> the TIME arm stops; nothing else is run. The weekend slice is +0.00021 +- 0.00051 (0.4 SE), the two seeds
disagree in sign by 0.0025 (one-seed weekend noise ~0.0007 SE each), the weekday slice is -0.00009, and the overall holdout / val deltas are
exact nulls (+0.00001 / +0.00017). The standard SCALE-UP verdict (information) is also FAIL. Reading: with the feed's weekend flag and hour
available as dense inputs, the 2-epoch recipe learns NO usable weekend x category (or x hour) interaction: the weekend-slice effect is
inside its SE and the hour inputs do not move val either (hour mix identical across splits). This agrees with the CURRENT_STATUS external-data
close (every calendar x category / x hour prior within +-0.0003 on val and on the weekend-containing holdout): the weekend category shift
(log-rate sd 0.046, within-feed spread 0.08) is too small to re-order candidates within a feed. Nothing built (no abl_e10_time_*, no
preds/ens_Q4time_*, no subs/cand_Q4_*), nothing submitted, no process of this agent left running. The feature files
data/feats/*/tm_{weekend,hsin,hcos}.npy (+ hard links) stay on disk for any later use. The 10-epoch family, FT, and the Q4 pool read were
pre-declared on PASS only and were NOT run; their scripts (nn_x/abl/time/launch_e10_time.sh, chain_time.sh, decide_time.py; decide_time.py
plumbing-tested with the pop2 members as stand-ins: it reproduces decide_x8 Q3's numbers exactly, weekend-slice SE 0.00038 at the pool level)
are kept for reference.

## 10. ENCODER-DOSE pre-declaration (written 2026-11-14T00:03:07Z (`date -u`), BEFORE any run of this task; reports/SOLVER_POWER_1113.md section 6.3 item 2 and section 3; agent; TIME CAP 8 h from 2026-11-13T23:51Z, i.e. until 2026-11-14T07:51Z)

State at writing: no abl_pre2noid_* / abl_e10_xdrop_pre2_* / pre2n_* / w5_e10_xdroppre2_* file exists anywhere under nn_x/abl,
data/feats, nn/feat_cache or nn_x/w5/preds (find: 0 matches). the GPUs all idle (0 MiB, 0 compute processes). Files at writing
(md5): abl_lane.sh 1c933012ecd51853a8469e09af0e3af5, train.py 027501d53edba2f57e9bd2e9a29e05aa, abl_arms.py
ed2fe77b118129592f652c528cd39556, abl_eval.py c79d9580a416678ee86f05639d327232 (abl_eval.py is NOT edited by this task).
Question (SOLVER_POWER 6.3 item 2): does encoder quality drive ranker value, and does a pretrained REPRESENTATION (not 6
scalars) add anything? Calibration point: the pre2 channel (6 cvnd_pre2s columns pre2_fe_z, pre2_new_z, pre2_fe_s, pre2_new_s,
pre2_hmax32, has:pre2_fe) is worth +0.00128 +- 0.00061 val / +0.00054 +- 0.00034 holdout at 2 epochs (section 6.2, LAST
checkpoints, abl_base_s23/24 vs abl_xdrop_pre2_s23/24, logs/abl_eval_wave2.txt) and +0.0009 val at 10 epochs (F9 vs F6).
Order of launches (independent reads, both pre-declared here): (B) is launched FIRST because it is the long pole (~95 min per run
alone) and its verdict does not depend on (A); (A) is launched once its columns are built and its flag is tested. The report
order stays (A), (B), (C).

### 10.A In-ranker dose: arm pre2noid (2-epoch screen, seeds 23 / 24, GPUs N)

| arm | type | exact change vs the base (A_BASE + HYG, seeds 23 / 24) | mechanism |
|---|---|---|---|
| pre2noid | SWAP (input substitution; read against BOTH base and xdrop_pre2 with the frozen section-4 machinery) | append `--xt_exclude pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32,has:pre2_fe --xt_add pre2n_fe_z,pre2n_new_z,pre2n_fe_s,pre2n_new_s,pre2n_hmax32,has:pre2n_fe --xt_add_relonly pre2n_fe_s,pre2n_new_s,pre2n_hmax32` (variants abl_pre2noid_s23 / s24; the --xt_exclude list is byte-identical to the xdrop_pre2 arm's) | the 6 live pre2 columns are the LAST 6 of the cvnd_pre2s preset (indices 28-33), so after exclude + add the 6 noid columns sit at exactly the same indices 28-33; the new flag --xt_add_relonly puts the 3 named added columns into xt_rel (after the kept XT_REL entries seq_cv_invd20, it_hist_all_luo -> the same index list as the base's) and into xt_relonly (the same 3 indices as the base's); xt_idcols unchanged; n_dense 95 = the base's. The model layout (xt_names count, xt_rel, xt_relonly, xt_idcols, n_dense, dense_features order) is therefore IDENTICAL to the base's; only the CONTENTS of the 6 columns change (ID-free encoder scores instead of ID-aware ones). attach_xt transforms by name suffix exactly as for the live columns: *_z clip +-5, _s / hmax32 clip +-10 (unclipped in practice), has: indicator. |

**Columns** (nn_x/abl/encdose/build_pre2noid.py; written to data/feats/{train,val,test}/pre2n_{fe,new,fe_z,new_z,fe_s,new_s,hmax32}.npy
by exclusive create and hard-linked into nn/feat_cache/{train,val,test}/, the pop1/build_xp_imp_c50.py pattern; nothing existing
is touched). They are the pre2_* construction (reports/pretrain_report.md 8c: pretrain/assemble.py for val / test with tags v3At2,
v4At2 calibrated to the test-side twins v3t2, v4t2 on val and tags v3t2, v4t2 on test; pretrain/assemble_train.py for train with
--wtag "w{k}r_{c}" --rtag "{c}t2", windows 1-3, configs v3 / v4) with the raw keys SWAPPED: fe -> fe_noid, new -> new_noid,
hmax32 -> hmax32_noid, where fe_noid / new_noid are the stored ID-zeroed head scores (preds/pre_<tag>_{fe,new}_noid_<split>.npy,
data/pretrain/winscores/pre_<tag>_{fe,new}_noid_win<k>[_s60000].npy; pretrain/train.py infer(): the same query, the candidate's
out-vector with the ID slot zeroed, no id_bias) and hmax32_noid is NEW: the infer() hmax32 arithmetic (max over the newest 32
history items of <out_vec(item_rep(cand)), out_vec(item_rep(hist))>, -2 when the history is empty, bf16 autocast item reps) with
the ID slot zeroed for the candidate AND the history items, scored from the same checkpoints (ckpt_w{1,2,3}r_{v3,v4} on their
windows, ckpt_{v3,v4}t2 on the 60K window samples and on val / test, ckpt_{v3A,v4A}t2 on val) by nn_x/abl/encdose/score_hmax_noid.py
(GPU N or 5, one process, <= 20 min; no transformer pass is needed because hmax32 uses item representations only). Same affine
moment matching (noid raw to the noid raw of the twin / reference sample, over finite values > -2), same within-feed z
(src/stack_val.feed_rank_z) for _z, same mean over the two configs, NaN outside the windows (the NaN mask equals has:pre2_fe's),
then _s = (x + 20) / 6 (the live pre2_fe_s / pre2_new_s transform, verified bit-exact on val: max |pre2_fe_s - (pre2_fe + 20) / 6| = 0).
pre2n_fe / pre2n_new (raw) are written for the has: indicator and for reference. Self-checks before the arm is launched:
(i) the builder re-run on the LIVE keys (fe, new, hmax32) must reproduce pre2_fe, pre2_fe_z, pre2_new, pre2_new_z, pre2_hmax32
(max |diff| reported; expected 0 or float rounding); (ii) score_hmax_noid.py in ID-AWARE mode must reproduce the stored hmax32 of
the same checkpoint (corr > 0.999 on val for v3At2; reported); (iii) parity PSI train-vs-val of pre2n_fe_s / pre2n_new_s /
pre2n_hmax32 < 0.5 on 10 train-quantile bins; (iv) NaN mask of pre2n_fe == NaN mask of pre2_fe on every split.
FALLBACK (pre-declared): if (ii) is not verified by 02:30 UTC, pre2n_hmax32 is written as a COPY of pre2_hmax32 (the ID-aware
column) and the arm is read as the 5-column swap; this is disclosed in 10.A results.

**Code** (every new flag defaults off and is bit-identical when off; train.py / abl_arms.py / abl_lane.sh replaced by write-new + mv):
- abl_arms.xt_add_relonly_idx(xt_rel, xt_relonly, xt_names, add_spec, spec, log): returns (xt_rel + idx, xt_relonly + idx) for the
  --xt_add columns named in spec; refuses a name not in --xt_add, a duplicate, or an index already in xt_rel / xt_relonly.
- train.py `--xt_add_relonly` (default ''), called once behind `if a.xt_add_relonly:` right after the xt_relonly lookup in the
  no-init branch; the --init refusal guard gains `or a.xt_add_relonly`.
- abl_lane.sh: arm pre2noid appended (nothing else changed; ALLOWED default unchanged, launched with ABL_GPUS="4 5").
- Tests: tests/test_abl.py cpu (ABL_FLAGS / OFF / the AST counts extended: 4 imports, 5 add_argument, 6 guards) must PASS
  110+/110+; nn_x/abl/encdose/test_encdose.py: flag-off identity of the helper, pre2noid layout == base layout (xt_rel, xt_relonly,
  xt_idcols, n_dense) with the 6 names swapped at indices 28-33, lane DRY command = base + the three flags, refusals. An
  ABL_SMOKE=1 plumbing run of the arm on GPU N (abl_smoke_pre2noid_s23, 20K feeds, 1 epoch) must reach rc 0 before the screen.

**Rule (A): KILL ARITHMETIC** (val, LAST checkpoints = step 19,670, per-feed paired NDCG deltas on the same seed, pooled over seeds
23 / 24 as the per-feed mean of the two per-seed deltas, then the mean over the 14,423 val feeds; SE_cal = sqrt(SE_feed^2 +
sigma_run^2 / 2), sigma_run 0.00035 val / 0.00015 holdout; the frozen section-4 convention, machinery = abl_eval.read_arm imported
read-only by nn_x/abl/encdose/abl_eval_encdose.py, which registers the arm in its own copy of ARMS and never edits abl_eval.py):
- b = abl_base_s<seed>@last, x = abl_xdrop_pre2_s<seed>@last, n = abl_pre2noid_s<seed>@last.
- W = mean(b - x) = the channel's worth (frozen read +0.00128 val; recomputed from the same prediction files, it must equal it).
- L = mean(b - n) = the value LOST by swapping the ID-aware channel for the ID-free one.
- f = L / W = the fraction of the channel's worth that the swap loses.
- **KILL encoder scaling iff f < 0.40, i.e. L < 0.40 x W = 0.40 x 0.00128 = +0.000512 val** (SOLVER_POWER 6.3: "< 0.0005 of
  +0.00128"). Equivalent statements: n - b > -0.000512 (the swapped arm sits within 40% of the channel's worth of the base), or
  n - x > 0.60 x W = +0.000768 (it sits closer to base than to xdrop_pre2 by more than 60 / 40). A swap that GAINS (L <= 0) kills.
  Meaning: the ID part of the encoder (0.004-0.005 zero-shot, the only encoder-quality lever we can switch) does not drive ranker
  value, so a better encoder is not expected to either.
- **NOT KILLED iff L >= 0.40 x W** (the swap loses at least 40% of the channel's worth). This read is ~1 SE at the bar (SE_cal(L)
  ~0.0006): it can CLOSE encoder scaling, it cannot CONFIRM it; NOT KILLED only keeps the escalation path open subject to (B) / (C).
- Decision on the point estimate; reported with it: SE_cal(L), z = (L - 0.40 W) / SE_cal(L), per-seed L, the holdout analogue
  (W_hold frozen +0.00054, bar 0.40 x W_hold = +0.000216; information), the selected-checkpoint read, val slices (M<50, M>=50,
  Tue, Thu, coldpos, warmpos), holdout late20%, pmix, levers, CLIPLOG, minutes, the standard SCALE-UP verdict text of both blocks
  (information only). Stale-val is NOT available for this arm: stale_pred.py overrides pre2_fe / pre2_fe_z / pre2_fe_s by NAME and
  the pre2n_* names have no counterfactual files; it is skipped and said so.
- Readout files: nn_x/abl/logs/abl_eval_encdose.txt (+ .json), written once by abl_eval_encdose.py after both runs are DONE.

### 10.B PRE2-10EP: abl_e10_xdrop_pre2_s951 / s952 vs w5_e10_base_s951 / s952 (GPUs N, one run each)

- Runs: `train.py --variant abl_e10_xdrop_pre2_s<seed> <E_BASE> <HYG> --seed <seed> --xt_exclude pre2_fe_z,pre2_new_z,pre2_fe_s,pre2_new_s,pre2_hmax32,has:pre2_fe`
  = exactly the nn_x/abl/pop1/launch_e10.sh command with the pop1 flags replaced by the xdrop_pre2 --xt_exclude
  (nn_x/abl/encdose/launch_e10_xdrop_pre2.sh; the logged args must equal abl_e10_pop1_s951's except variant / seed / the arm flags;
  no --no_test, so trainonly_test preds are written too). s951 on GPU N, s952 on GPU N. Reference = the existing w5_e10_base_s951 /
  s952 (nn_x/w5/preds; nn_x/abl is bit-identical to nn_x/w5 at flags off, tests/test_abl.py). Nothing is copied into nn_x/w5/preds:
  the read-out reads the arm from nn_x/abl/preds and the base from nn_x/w5/preds directly.
- Read-out nn_x/abl/encdose/readout_pre2_10ep.py = the w5_eval_e_ap1.py mechanics (selected checkpoints PRIMARY, @last information;
  per seed; per-feed paired deltas pooled over the two seeds; val slices M<50, M>=50, Tue, Thu; pmix; the 2-member pool) with the
  arm path redirected and the SIGN turned into the channel's WORTH: Wv = mean(base - xdrop_pre2) on val, Wh on the holdout,
  W_M50 on the val M>=50 slice. Plus, per the CURRENT_STATUS 11-13 16:10 amendment (b), the holdout late (Sep 17-19) and early
  (Sep 6-8) columns as information. Plumbing check before the runs finish: run on w5_e10_ap1 vs w5_e10_base (sign flipped) and
  reproduce nn_x/w5/logs/w5_eval_E_ap1.txt to the printed digits.
- **SEQUENTIAL GATE (SOLVER_POWER 6.3 (B), verbatim: "Sequential gate, not a parallel read (GLOBAL_RECAP section 5 row 7):
  >= +0.0015 val, holdout > 0, M>=50 > 0"), applied in this order:**
  (1) Wv >= +0.0015 (pooled val worth, selected checkpoints). If FALSE: STOP, verdict FAIL; clauses (2) / (3) are printed but do
      not count.
  (2) Wh > 0 (pooled holdout worth). If FALSE: FAIL.
  (3) W_M50 > 0 (pooled val M>=50 slice worth). If FALSE: FAIL.
  PASS iff (1), (2) and (3) hold in sequence.
- Consequence: PASS -> the pre2 channel reaches the E1 / F8 trigger at 10 epochs; it counts as "(B) passes" in the escalation rule
  below and is reported to the lead (PRE2-XL re-open information; GLOBAL_RECAP section 5 row 7 "only after PRE2-10EP fires");
  nothing is launched by this agent. FAIL -> PRE2-10EP closes; the two runs stay as the pre2-free 10-epoch reference for
  PSEUDO-DIRECT (SOLVER_POWER 6.3 item 3). Information: the 10-epoch worth vs the 2-epoch worth (+0.00128) and vs F9 - F6 (+0.0009).
- Readout files: nn_x/abl/logs/readout_pre2_10ep.txt (+ .json), written once after both runs are DONE.

### 10.C Frozen hidden-state K/V probe (design and pass rule, SOLVER_POWER 6.3 (C) verbatim; run only if time allows within the cap)

"One frozen checkpoint for every row (`ckpt_w1r_v3`, corpus ends Sep 11), run ID-zeroed (trained with ID-dropout 0.3). Export 64-d
last-layer states with one fixed PCA for train rows Sep 12-19, the holdout, val and test; earlier rows get a learned null token. One
latent space, so there is no train/serve misalignment and no later-snapshot leak. 2-epoch pair: abl_base vs abl_base + a zero-init
gated K/V adapter in the 2 cross-attention blocks. Stream states from pinned host RAM, at most 2 processes per GPU. **GO** only if
val >= +0.0015 with holdout > 0 and stale-val / pmix vetoes >= -1 SE."
Design as implemented or skeletoned here (nn_x/abl/encdose/kv_probe/): (1) export_states.py: for every feed of train (Sep 12-19
only, day >= 20708 MSK), the 30K holdout feeds (all of them are train feeds; those on Sep 6-11 get the null token by the leak
rule), val and test, run SeqEncoder(ckpt_w1r_v3) with the ID slot zeroed over the reversed history (old -> new), take the LAST-LAYER
states h[1..H] (one per history event, d 384) and the user state h[H]; fit ONE PCA (384 -> 64) on a fixed 2M-state sample of the
Sep 12-14 train rows (seed 0), apply it everywhere, store fp16 per-event states aligned with hist_off / H plus a per-feed flag
(1 = exported, 0 = null token) in data/feats/<split>/kv64_* (exclusive create). Leak rule: a train / holdout feed on or before Sep 11
(the corpus end of ckpt_w1r_v3, which was pre-trained on those snapshots) gets NO states (flag 0, the learned null token);
val / test are after the corpus end by construction. (2) kv_adapter.py: a zero-init gated K/V branch for the 2 cross-attention blocks:
K' = [K; Wk s], V' = [V; Wv s] over the exported 64-d states (masked to the feed's own history), output scaled by tanh(g) with g
initialised to 0 -> the ranker is BIT-IDENTICAL to the base at initialisation and with the flag off (unit test). (3) 2-epoch pair
abl_base_s23/24 (existing) vs abl_kv_s23/24 (A_BASE + HYG + --kv64). Pass rule verbatim: GO only if pooled val delta >= +0.0015
(LAST checkpoints, the frozen section-4 machinery) with pooled holdout delta > 0 and stale-val delta >= -1 SE and pmix delta >= -1 SE.
STATUS RULE: if the pair cannot be launched and read within the cap, (C) is reported as PENDING with the design, the code skeleton
and its tests; no number of (C) is read before its pre-declared rule is re-confirmed by the lead.

### 10.D Escalation rule (SOLVER_POWER 6.3 (C) row "escalation", verbatim)

"only if (C) passes AND either (A) shows the swap losing >= 40% of the channel's worth or (B) passes: Encoder sweep 30M / 90M, 2
encoder seeds per size, in one latent space: a warm-start chain with a Procrustes R^2 >= 0.95 anchor check, or a single encoder on a
corpus that precedes the scored rows. Gate on the nested residual on val and the holdout, and on cell AUC above the 0.570 / 0.632
controls. Then a 10-epoch pair and a family: 70-90 GPU-h in total." The escalation is the lead's decision; this agent launches nothing
beyond 10.A / 10.B / (10.C if time allows), fine-tunes nothing, pools nothing, submits nothing.

**GPUs / process rules:** only GPUs N and 5; nvidia-smi before EVERY launch (<= 3 compute processes and >= 60 GB free on that GPU);
CUDA_VISIBLE_DEVICES = one id; OMP / MKL / OPENBLAS / NUMBA / POLARS threads 12; setsid nohup ... < /dev/null &; the train.py /
score PIDs verified through /proc/<pid>/cmdline and recorded in nn_x/abl/RUNNING.txt; stop only this agent's processes by exact PID;
never pkill -f / kill $(pgrep); nothing of any other process is touched; lane scripts edited only by write-new + mv; no project data
or user identity leaves the machine; reports/ and nn_x/w5/* are not edited.

## 11. SSL-WIDTH pre-declaration (written 2026-11-14T00:09:48Z (`date -u`), BEFORE the first number of the E2 trigger and before any Step-0 run; reports/SOLVER_POWER_1113.md section 6.3 item 4 and sections 3, 5; reports/PRETRAIN_PLAN_1113.md section 5 row E2; agent; TIME CAP 8 h from 2026-11-13T23:51Z)

State at writing: no recall, no per-feed G3 NDCG and no correlation has been computed. nn_x/w11/ssl/trigger/out/ is empty. Design-time
reconnaissance only (nn_x/w11/ssl/trigger/e2_trigger.py written but not run; scratch recon_align.py): feed counts per corpus status,
the window-corpus reconstruction check against data/pretrain/win{2,3}/info.json (sequence and token counts exact; ID rows 34,782,939 /
41,731,192 = the checkpoints' tables), and the finding that same-user snapshots are NOT exact copies of one event stream (the later view
lacks some of the earlier view's newest items), so "events newer than the user's corpus snapshot" cannot be recovered by alignment.

### 11.1 Step -1: the E2 reopen trigger (CPU, 0 GPU-h)

**Trigger as written (PRETRAIN_PLAN section 5, E2 row; SOLVER_POWER 6.3 item 4):** "partial corr > 0.05 at 2 SE between the per-feed G3
NDCG residual and the hidden-event recall@10 of the windowed w3r encoder on post-window holdout feeds, controlling for M, H and cold share.
FAIL -> close GENREC-JOINT, JOINT-SSL and SSL-as-capacity-enabler together at 0 GPU-h."

**Deviation, disclosed before the first number.** "Post-window holdout feeds" = the 8,279 holdout feeds with timestamp >= 2026-09-17 00:00
MSK (the 'late' holdout column of the 16:10 rule amendment; W3 starts there). The w3r corpus (build_corpus_window.py) contains the
EARLIEST W3 snapshot of every non-zero user and every user-0 W3 feed, so on this slice w3r was trained on the hidden events of 7,319 of
the 7,847 non-zero-user feeds (status insample_self) and of all 432 user-0 feeds: the literal w3r reading is an in-sample probe, exactly
the defect the trigger was written to remove ("the comparator ... was trained in-sample on the same val-snapshot targets"); and the
remaining 528 feeds are too few for a 2-SE read at 0.05 (SE ~0.05). The nearest honest instance of the defined trigger is the SAME
encoder family one window earlier: w2r_v3 (corpus = every sequence before 2026-09-17 MSK; val fe 0.3538 vs w3r 0.3566), for which every
late feed is strictly post-window, evaluated on the late feeds whose user has NO sequence in the w2r corpus (no train feed before Sep 17;
6,000 feeds, 5,191 with a target). The w3r columns are reported as information.

**PRIMARY (the verdict):**
- encoder: ckpt_w2r_v3.pt (d 384, 4 blocks, ID table 34.8M x 64, item_tab2), fe head, IDs on (the deployed 'fe' score); CPU fp32.
- feeds: late holdout slice (ts >= Sep 17 MSK) AND user_id != 0 AND the user has no w2r corpus sequence (status nocorpus) AND >= 1 target.
- hidden events = the newest min(16, H) history events; prefix = the older events (BOS state if empty); targets = distinct hidden items
  with event_type 1 and event_source 1 that do not occur in the prefix (the encoder's own fe-target rule).
- fixed universe = items with >= 1 fe-target occurrence in the w2r corpus (build_corpus.targets rule) UNION all targets of the slice;
  ranking = fe-head cosine logit of the prefix state against every universe item, the feed's prefix items removed;
  recall@10 = |targets in the top 10| / |targets|.
- G3 per-feed NDCG = member-equal z-mean of the 20 train-only members (v23_gru_pre2s_s501-506, r6_cd_m02_r101-107,
  r6_cd_m02_p70k_r101-107; preds/holdout/) with holdout_y, per feed (src/metrics.ndcg_per_feed).
- controls Z = [1, M, log(1+M), H, log(1+H), cold_share], cold_share = share of the feed's candidates with it_hist_all_luo == 0
  (LUO-cold, the ranker's own exposure input). Partial corr = Pearson corr of the OLS residuals of NDCG and recall@10 on Z.
- SE = user-cluster bootstrap over feeds (2,000 resamples, seed 0); the Fisher 1/sqrt(n-k-3) SE is information.
- **PASS iff partial corr - 2 SE_boot > 0.05** ("> 0.05 at 2 SE" read as the lower 2-SE bound above 0.05). The lenient reading (r > 0.05 and
  r > 2 SE) is printed as information and is NOT the gate. One look.
- **FAIL -> GENREC-JOINT, JOINT-SSL and SSL-as-capacity-enabler are CLOSED together at 0 GPU-h; nothing of Step 0 runs.**

**Information columns (never gates):** recall@50; K = 8 hidden events; extended controls (+ log(1+n_targets), log(1+prefix length));
all 7,105 w2r-evaluable late feeds (incl. user0 and users with a corpus snapshot); user0 only; hascorpus only; w3r_v3 on all late feeds
(the literal, in-sample reading) and on its insample_self / user0 / hascorpus subsets; raw (unpartialled) correlation; Spearman of the
residuals; mean G3 residual at recall = 0 vs > 0. Script: nn_x/w11/ssl/trigger/e2_trigger.py (score w2r_v3 | score w3r_v3 | readout);
outputs nn_x/w11/ssl/trigger/out/{recall_*.npz, readout.txt, readout.json}.

### 11.2 Step 0 on PASS only: the 2x2 {d128, d192} x {aux off, aux on}, seeds 23 / 24 (GPUs N and 7)

Code: private copy nn_x/w11/ssl/ of nn_x/abl (the harness that runs the 2-epoch A-stage screen; abl_base_s23/24 are the d128-off cell),
one flag for the auxiliary head, OFF by default and bit-identical when off (AST proof + CPU end-to-end identity, the nn_x/w9/catgrad
test pattern), --d 192 for the width cells (the flag exists). Aux (SOLVER_POWER 6.3 item 4, verbatim intent): an ID-free next-fe sampled
softmax with logQ over in-batch history items only; no current-feed candidates as negatives; targets at a gap of 16-64 events from the
state, over all fe&new positions; weight 0.1 (0.01 as information if time allows); causality: the transformer history encoder uses a
newest-first key mask key_idx > query_idx with no user token; unit test: perturbing event j leaves the aux state at j+1 bit-identical.
Runs: abl-style variants on A_BASE + HYG, seeds 23 / 24: d128+aux (2), d192 (2), d192+aux (2); d128 = abl_base_s23/24 (existing).
**Kill rule (verbatim):** d128+aux < -1 SE val, OR interaction (d192+aux - d128+aux) - (d192 - d128) < +0.001 val at 2 SE, OR holdout
sign disagrees; read at the last step, the matched plateau step as information. ("< +0.001 at 2 SE" read as: interaction - 2 SE < +0.001
-> kill; SE from the per-feed paired deltas pooled over the two seeds with SE_cal, sigma_run 0.00035 val / 0.00015 holdout.)
Escalation (on a non-kill, NOT run inside this task): a 10-epoch 4-seed pair at a matched epoch count; level rule member >= +0.002 val,
>= +0.0014 holdout, in-pool replay >= +0.0005. Nothing fine-tuned, pooled or submitted by this task.
GPU rules: only GPUs N; nvidia-smi before every launch (<= 3 compute processes, >= 60 GB free); CUDA_VISIBLE_DEVICES = one id; 12 CPU
threads; setsid nohup; real PIDs via /proc/<pid>/cmdline in nn_x/abl/RUNNING.txt; stop only by exact PID; lane scripts write-new + mv.

### 11.3 E2 TRIGGER RESULT (written 2026-11-14T00:38:56Z; read out 00:43 UTC by the pre-declared rule of 11.1; nn_x/w11/ssl/trigger/out/readout.txt + .json, first-look copy readout_first_look_w2r_only.txt; scoring logs out/score_w2r_v3.log (26.6 min CPU, 12 threads, PID 2505116) and out/score_w3r_v3.log)

Reconstruction checks: the w2r corpus rebuilt from data/flat/train matches win2/info.json exactly (2,125,706 sequences, 389,087,050 tokens,
34,782,939 ID items = the checkpoint's table); fe-target universe 40,753,124 corpus items + 16,698 target-only items (K 16). Late slice
8,279 feeds: nocorpus 6,000 / hascorpus 1,847 / user0 432; with >= 1 fe target among the newest 16 events: 5,191 / 1,680 / 235.
Floor effect (disclosed): over the 40.8M-item universe the fe head puts a target in the top 10 for only 4.3% of the primary feeds (mean
recall@10 0.0070; recall@50 0.0237, nonzero 13.0%), so recall@10 is close to a rare binary indicator; the pre-declared rule is applied as
written and the graded information columns (recall@50, Spearman of the residuals) are reported next to it.

| column | n feeds / users | partial corr +- SE_boot (Fisher) | lower 2-SE bound | z | raw corr | Spearman(partial) | G3 residual at recall = 0 / > 0 |
|---|---|---|---|---|---|---|---|
| **PRIMARY: w2r_v3, late feeds of users absent from the corpus, K 16, recall@10, controls M, log M, H, log H, cold share** | **5,191 / 5,184** | **-0.0012 +- 0.0139 (0.0139)** | **-0.0290** | **-0.09** | -0.0028 | +0.0045 | +0.0003 / -0.0068 |
| same, recall@50 (info) | 5,191 | +0.0229 +- 0.0160 | -0.0091 | +1.43 | +0.0309 | +0.0519 | -0.0016 / +0.0105 |
| same, K 8, recall@10 (info) | 4,615 / 4,610 | -0.0287 +- 0.0109 | -0.0504 | -2.64 | -0.0213 | +0.0680 | +0.0011 / -0.0362 |
| same, + controls log(1+n_targets), log(1+prefix length) (info) | 5,191 | -0.0013 +- 0.0140 | -0.0293 | -0.09 | -0.0028 | +0.0014 | +0.0005 / -0.0106 |
| all 7,106 w2r-evaluable late feeds incl. user0 + hascorpus (info) | 7,106 / 6,864 | +0.0047 +- 0.0121 | -0.0196 | +0.39 | +0.0027 | -0.0030 | -0.0001 / +0.0026 |
| hascorpus only (hidden events possibly in-sample) (info) | 1,680 / 1,679 | +0.0084 +- 0.0262 | -0.0440 | +0.32 | +0.0077 | +0.0052 | -0.0004 / +0.0115 |
| user0 only (info; first-look SE invalid: one bootstrap cluster; re-read with per-feed clusters in 11.4) | 235 | +0.1264 (Fisher 0.0664) | - | - | +0.0873 | +0.0151 | -0.0067 / +0.1363 |

**VERDICT (pre-declared rule 11.1, strict reading r - 2 SE > 0.05): FAIL.** The lenient reading (r > 0.05 and r > 2 SE) also FAILS; the point
estimate is zero to three decimals and the K-8 column is slightly NEGATIVE (z -2.6): feeds whose hidden recent events the encoder can
predict are, if anything, feeds where G3 does slightly worse than its M / H / cold-share expectation. No information column reaches
+0.05 at any SE. **Consequence (pre-declared): GENREC-JOINT, JOINT-SSL and SSL-as-capacity-enabler are CLOSED together at 0 GPU-h.**
Step 0 (the 2x2 {d128, d192} x {aux off, aux on}) does NOT run; nn_x/w11/ssl/ holds only the trigger code (no harness copy, no flag,
no GPU process of this task was ever started). Nothing fine-tuned, pooled or submitted. The w3r_v3 information columns (the literal,
in-sample reading of the trigger) are appended as 11.4 when its CPU scoring finishes (information only; they cannot change the verdict).
# 2026-11-14T00:39:14Z [SSL-WIDTH agent] correction to 11.3: the primary readout ran at the time stamped in out/readout_first_look_w2r_only.txt's header (00:38 UTC), not 00:43; the ledger section 11.3 was written 00:38:56Z immediately after it

### 11.4 E2 TRIGGER information columns, w3r_v3 and the user0 re-read (written 2026-11-14T00:49:14Z; final readout 00:49 UTC, out/readout.txt + .json; score_w3r_v3.log 30.3 min CPU, PID 2524696; information only, the verdict of 11.3 stands)

w3r corpus rebuilt = win3/info.json exactly (2,706,012 sequences, 491,885,750 tokens, 41,731,192 ID items); universe 47,069,689 items.
Late slice under w3r: insample_self 7,751 (every non-zero user's earliest W3 snapshot AND every user-0 W3 feed are training sequences),
hascorpus 528, nocorpus 0: the literal w3r reading has no honest subset of useful size, as stated in 11.1.

| column (K 16, recall@10, the 11.1 controls) | n feeds / users | partial corr +- SE_boot | lower 2-SE | z | raw | Spearman(partial) | recall@10 mean (nonzero) |
|---|---|---|---|---|---|---|---|
| w3r_v3, ALL late feeds with a target (the literal, IN-SAMPLE reading) | 7,106 / 6,864 | -0.0006 +- 0.0121 | -0.0249 | -0.05 | +0.0063 | +0.0843 | 0.0226 (9.4%) |
| w3r_v3, insample_self only | 6,631 / 6,397 | -0.0040 +- 0.0122 | -0.0285 | -0.33 | +0.0033 | +0.0827 | 0.0236 (9.7%) |
| w3r_v3, hascorpus only (528 feeds, 475 with a target) | 475 / 474 | +0.0904 +- 0.1055 | -0.1205 | +0.86 | +0.0843 | +0.0728 | 0.0079 (5.1%) |
| w2r_v3, user0 only, re-read with one bootstrap cluster per feed (corrects the 11.3 row) | 235 | +0.1264 +- 0.0752 | -0.0240 | +1.68 | +0.0873 | +0.0151 | 0.0080 (4.7%) |

Reading: in-sample training raises the encoder's hidden-event recall 2-3x (9.4% vs 4.3% nonzero) and still leaves the Pearson partial
correlation with the G3 residual at zero (-0.0006 +- 0.0121); the rank version (+0.08) is the only positive signal anywhere and it is
in-sample. Nothing in 11.3 / 11.4 reaches the +0.05 bar at any SE. **CLOSED: GENREC-JOINT, JOINT-SSL, SSL-as-capacity-enabler (0 GPU-h).**
Processes of this task: PIDs 2505116 and 2524696 (CPU only), both ended rc 0 (RUNNING.txt); no GPU was used; nothing under nn_x/abl
except ledger.md / RUNNING.txt and nothing under nn_x/w5 or reports/ was edited; code and outputs: nn_x/w11/ssl/trigger/.

### 10.A RESULTS: in-ranker dose (written 2026-11-14T00:51:46Z; read out 00:50:0x UTC by the pre-declared rule; nn_x/abl/logs/abl_eval_encdose.txt + .json)

Build (all before the arm's launch): hmax32_noid scorer verified in ID-AWARE mode against the stored preds/pre_v3At2_hmax32_val.npy
(max |diff| 1.4e-4, corr 1.000000, sentinel agreement 1.0; logs/encdose_hmax_verify.log); the 18 hmax32_noid scorings took 80 s on GPU N
(logs/encdose_hmax_noid.log; the window corpora's tok_*.npy no longer exist, so the item categories come from the full corpus tables:
the history items of window feeds are in both corpora and candidates take their split's category, so nothing differs for the scored
rows). build_pre2noid.py --check reproduced the live pre2_fe / pre2_fe_z / pre2_new / pre2_new_z / pre2_hmax32 EXACTLY on train / val /
test (max |diff| 0, same NaN masks; logs/encdose_build_check.log). The 21 pre2n_* files (7 per split) were written at 00:16 UTC by
exclusive create with hard links into nn/feat_cache (encdose/out/pre2noid_parity.json). The FALLBACK clause was NOT needed.
Parity / information: NaN mask == pre2_fe's on every split (train 27.95% NaN outside the windows); PSI train-vs-val fe_s 0.134, new_s
0.124, hmax32 0.126, fe_z 0.011, new_z 0.006 (all < 0.5; train-vs-test <= 0.124); corr(pre2n, pre2) fe 0.931 train / 0.969 val / 0.972
test, new 0.943 / 0.973 / 0.975, hmax32 0.992 / 0.994 / 0.991. Zero-shot val NDCG: pre2n_fe 0.35367 vs pre2_fe 0.35780 (-0.0041, the
expected 0.004-0.005 ID share), pre2n_new 0.35614 vs 0.35916 (-0.0030), pre2n_hmax32 0.34842 vs 0.34551 (+0.0029: the ID-free
similarity is slightly better alone). Code: train.py --xt_add_relonly (abl_arms.xt_add_relonly_idx), lane arm pre2noid (md5s in
10.A results json / RUNNING.txt); tests/test_abl.py cpu 114/114 PASS (4 new checks; AST proof 4 imports / 5 add_argument / 6 guards; 117/117 after the (C) --kv64 wiring: 6 imports / 6 add_argument / 7 guards),
encdose/test_encdose.py 14/14 PASS (layout == the base's: xt_rel [0, 3, 30, 31, 32], xt_relonly [30, 31, 32], n_dense 104 = N_DENSE 61
+ 2 + 34 + 10 - 3 for both; the "95" in the 10.A pre-declaration was a transcription slip of the time-arm ledger, the equality claim is
what was pre-declared and it holds); ABL_SMOKE run abl_smoke_pre2noid_s23 rc 0 (00:19-00:21 UTC, GPU N).
Runs: abl_pre2noid_s23 (GPU N, PID 2537594) and abl_pre2noid_s24 (GPU N, PID 2537640), launched 00:22:02 UTC by abl_lane.sh (1 other
compute process on each GPU: the (B) run), 26.3 / 26.5 min, rc 0; log lines 'abl xt_add_relonly: rank/z-only for the added
[pre2n_fe_s, pre2n_new_s, pre2n_hmax32] (indices [30, 31, 32]; xt_rel [0, 3] -> [0, 3, 30, 31, 32]; xt_relonly [] -> [30, 31, 32])'.
Selected = last (19,670) for s23, 19,000 for s24. @last val 0.39286 / 0.39266, holdout 0.31755 / 0.31754 (base 0.39364 / 0.39348,
0.31754 / 0.31754; xdrop_pre2 0.39214 / 0.39243, 0.31743 / 0.31655).

| quantity (LAST, pooled s23 / s24) | val | holdout |
|---|---|---|
| W = b - x (channel worth; frozen read reproduced exactly) | **+0.00128** +- 0.00061 (seeds +0.00150 / +0.00106) | +0.00054 +- 0.00034 |
| L = b - n (loss of the ID-free swap) | **+0.00080** +- 0.00052 (seeds +0.00078 / +0.00082; z_cal -1.56) | -0.00001 +- 0.00029 (seeds -0.00001 / -0.00001) |
| f = L / W | **+0.63** (bar 0.40) | -0.02 (information) |
| bar 0.40 x W_frozen | +0.000512 | +0.000216 |
| z = (L - 0.40 W) / SE_cal(L) | +0.56 | - |
| n - x (pre2noid vs xdrop_pre2) | +0.00048 +- 0.00058 (seeds +0.00072 / +0.00024; bar 0.6 W = +0.00077) | +0.00055 +- 0.00032 (z_cal +1.73) |
| selected-checkpoint read (info) | L +0.00081 | L -0.00002 |
| val slices of n - b (info) | M<50 -0.00090, M>=50 -0.00073, Tue -0.00070, Thu -0.00092, coldpos -0.00027, warmpos -0.00110 (none < -2 SE) | late20% +0.00002 |
| pmix (info) | n - b -0.00106 +- 0.00067 (z -1.58); n - x +0.00032 +- 0.00082 | - |
| levers (info) | n - b: warm\|novel -0.00046 +- 0.00015, warm\|recent -0.00019, warm\|mid -0.00019; cold classes ~0 | - |
| stale-val | not available for pre2n_* (no counterfactual files), skipped as pre-declared | - |
| CLIPLOG | plateau clipped 0.000 on all runs | - |

**VERDICT (A) by the pre-declared rule: NOT KILLED.** The ID-free swap loses 63% of the channel's val worth (L +0.00080 vs the bar
+0.00051; z +0.56 above the bar), so encoder scaling is not closed by this read. Reading: on val the ID part of the encoder (0.004 of
zero-shot NDCG) carries ~60% of the channel's value inside the ranker, which is the direction expected if zero-shot quality drives
ranker value; but the read is 1 SE from the bar (it could not confirm, as pre-declared), the HOLDOUT shows the opposite (the ID-free
channel is worth exactly as much as the ID-aware one there: L_hold -0.00001; f -0.02; the holdout analogue 'would KILL'), and the
pre2noid-vs-xdrop_pre2 holdout delta (+0.00055, z_cal +1.7) equals the whole channel's holdout worth. The val loss sits in the
warm\|novel / warm classes (the ID table is a warm-item memory), consistent with the DATA_PIPELINE finding that the holdout is
popularity-regime-biased. Consequence per 10.D: the escalation stays conditional on (C) passing AND ((A) >= 40% [met on val] OR (B)
passing). Nothing further is launched from (A).

### 10.C process amendment (written 2026-11-14T00:52:38Z, BEFORE the first abl_kv64 screen run; rule, pass bar and read-out unchanged)

- Variant names: abl_kv64_s23 / s24 (lane arm kv64 = A_BASE + HYG + `--kv64 1`; the 10.C text said abl_kv_s23/24).
- The 64-d states are copied to the GPU of the training process (train: 272.8M rows x 64 fp16 = 34.9 GB, 1.47M exported feeds of
  3.15M; val 2.5M rows; the (GPU memory size removed) GPU holds them next to the ~50 GB training footprint) instead of being streamed from pinned host
  RAM; the gather is the ranker's own history index arithmetic (hist_off + k, newest-first) on the compact offsets kv64_off
  (kv_probe/kv_data.attach_kv64). Function identical, process simpler.
- Export done (kv_probe/export_states.py; logs/encdose_kv_fitpca.log, encdose_kv_export_{valtest,train}.log): PCA fitted on 2,813,997
  states of 15,000 Sep 12-14 train feeds (seed 0; 64 components explain 84.8% of the variance); val 14,423 / test 28,595 feeds all
  exported; train 1,470,065 feeds (Moscow day >= 20708 = Sep 12; the 1,681,143 feeds on or before Sep 11 get kv64_off -1 = the learned
  null token); 177 s on GPU N. Files data/feats/<split>/kv64_{states,off,user}.npy (exclusive create). Export smoke on 3,000 val feeds:
  slot-0 state == the user state h[H] on every checked feed (the newest-first mapping slot k = h[H - k] holds).
- Adapter: kv_probe/kv_adapter.py (per cross block i: ctx + tanh(g_i) * A_i(LN64(s)); g_i = 0 at init; learned user-slot and null
  vectors; parameters under model.extra_init seed 7368787 AFTER the Ranker, so the base init and the batch sequence are unchanged);
  wired by forward pre-hooks (model.py / data.py untouched); cfg['kv64'] = 1 recorded (Ranker(**cfg) then fails loudly, like retr_cat;
  kv_adapter.load_ranker_kv64 re-attaches). Tests kv_probe/test_kv_probe.py 11/11 PASS: leak rule, slot mapping, gather wrapper on real
  train batches with synthetic states, BIT-IDENTICAL logits with the adapter at init (eval and train mode), learning path / null token,
  kv64 checkpoint round trip, PCA; tests/test_abl.py cpu 117/117 PASS with the --kv64 wiring (AST flag-off proof: 6 imports,
  6 add_argument, 7 guards).
- Read-out: kv_probe/abl_eval_kv64.py (abl_eval.read_arm, LAST checkpoints, kv64 vs base, SE_cal) prints the pre-declared rule
  "GO only if val >= +0.0015 with holdout > 0 and stale-val / pmix vetoes >= -1 SE". The STALE-VAL veto cannot be produced by
  nn_x/abl/stale_pred.py for a kv64 checkpoint (it needs the adapter and the states); as for the time arm, an unavailable veto means
  NO GO can be declared. If val / holdout / pmix all pass, the stale read is listed as the one pending item (a kv64-aware stale_pred,
  ~1 h of code) and the verdict stays NO-GO until it exists. The holdout read of the kv64 runs is an in-period read; note that 26% of the
  holdout feeds are Sep 6-11 rows and therefore carry the null token (the leak rule), i.e. the holdout measures the adapter partly
  through its null path.
- GPUs N (one kv64 run each, launched by abl_lane.sh after the ABL_SMOKE run abl_smoke_kv64_s23 reaches rc 0 with holdout preds),
  <= 3 compute processes and >= 60 GB free checked by the lane; the (B) runs share the GPUs until ~01:30 UTC.
- (2026-11-14T00:57:23Z, while the abl_kv64_s23 / s24 screen runs are training; rule unchanged) The stale-val veto of 10.C WILL be produced:
  kv_probe/stale_pred_kv64.py = nn_x/abl/stale_pred.py's 'both_pre2both' scenario (its module imported: same feat_path override, same
  counterfactual files) with the model rebuilt by kv_adapter.load_ranker_kv64 and the val states attached; the kv states themselves have
  no counterfactual (they keep their val values, as the time arm's tm_* inputs did). Plumbing check on the smoke checkpoint
  abl_smoke_kv64_s23@last (base-reproduction corr vs its stored val preds) before the real read; then selected + @last of both seeds on
  GPU N; abl_eval_kv64.py reads the stale block (arm flag stale = True) and evaluates all four clauses of the pre-declared rule.

### 10.C RESULTS: frozen hidden-state K/V probe (written 2026-11-14T01:26:38Z; read out 01:25-01:27 UTC by the pre-declared rule; nn_x/abl/logs/abl_eval_kv64.txt + .json)

Runs: abl_kv64_s23 (GPU N, PID 2581335) and abl_kv64_s24 (GPU N, PID 2581376) = A_BASE + HYG + `--kv64 1`, launched 00:55:52 UTC by
abl_lane.sh after the ABL_SMOKE run abl_smoke_kv64_s23 (rc 0, 00:53-00:55; a first smoke at 00:51 failed before training on a CPU / GPU
device mismatch of the freshly built adapter, fixed by placing it on the model's device; its markers were cleared and nothing else
changed), 27.3 / 27.0 min, rc 0 (1 other compute process per GPU: the (B) run). Log lines: 'kv64: attached train states (272803322, 64)
fp16, exported feeds 1470065 of 3151208', 'attached val states (2499548, 64)', 'zero-init gated K/V adapter attached to 2 cross blocks
(16898 parameters; gates [0.0, 0.0]; bit-identical to the base at init)'. Selected = last (19,670) for both seeds. @last val 0.39316 /
0.39382, holdout 0.31767 / 0.31750 (base 0.39364 / 0.39348, 0.31754 / 0.31754). The adapter DID learn: gates tanh(g) at @last
[-0.33, +0.40] (s23) / [+0.33, +0.38] (s24), proj weight norms ~10, null / user0 norms ~1.1 (from 0), i.e. the states were used, they
just did not help. Stale-val produced by kv_probe/stale_pred_kv64.py (base-reproduction check corr 1.0000000, max |d| 0 on the first
variant; 8-11 s per checkpoint, GPU N).

| kv64 - base (LAST, pooled s23 / s24) | value | rule clause |
|---|---|---|
| val | **-0.00007** +- 0.00053 (z_cal -0.14; seeds -0.00048 / +0.00034) | >= +0.0015: **NO** |
| holdout | +0.00005 +- 0.00030 (z_cal +0.16; seeds +0.00013 / -0.00003); late20% -0.00052 | > 0: OK |
| pmix | -0.00030 +- 0.00071 (z -0.42) | >= -1 SE: OK |
| stale-val | -0.00061 +- 0.00047 (z -1.28); selected -0.00054 (z -1.14); PRIVATE TRADE -0.00053 | >= -1 SE: **NO** |
| val slices (info) | M<50 +0.00037, M>=50 -0.00038, Tue +0.00020, Thu -0.00038, coldpos -0.00053, warmpos +0.00005 | - |
| selected read (info) | val +0.00001 +- 0.00048, holdout +0.00004; coldpos +0.00146 (z +1.5), warmpos -0.00067 | - |
| levers (info) | warm\|recent +0.00042 +- 0.00037, warm\|mid -0.00026, warm\|novel -0.00019, cat#1\|w3+ +0.00022; total -0.00015 | - |
| minutes | 27.2 vs base 37.3 (ratio 0.73; load, not cost) | - |

**VERDICT (C) by the pre-declared rule: NO-GO** (val -0.00007, 3 SE below the +0.0015 bar; the stale-val veto also fires at -1.3 SE).
Reading: a frozen 64-d PCA representation of the pre2 encoder's per-event states, offered to both cross-attention blocks through a
zero-init gated adapter that the ranker was free to open (it did, to tanh(g) ~0.35-0.40), adds NOTHING to the 6 scalars on val, the
holdout, pmix or the private-like stale scenario (where it loses -0.0006). This is the "frozen hidden states as an extra K/V branch"
row of SOLVER_POWER section 3.4 (prior ~0.002-0.003): the pretrained representation carries no information the ranker's own history
encoder + the 6 scalars do not already have. Together with (A) (the ID-free swap keeps the whole holdout worth and 37% of the val worth)
the picture is that the encoder channel's value is a popularity-correlated scalar, not a representation.
Consequence per 10.D: (C) FAILS, so the escalation (encoder sweep 30M / 90M, 70-90 GPU-h) is OFF regardless of (A) / (B). Nothing
further is launched. Files kept: data/feats/{train,val,test}/kv64_{states,off,user}.npy (35.3 GB; may be deleted by the lead if space
is needed: the export reproduces in ~4 min), kv_probe/out/pca64.npz, the kv64 checkpoints / preds under nn_x/abl.

### 10.B RESULTS: PRE2-10EP (written 2026-11-14T01:31:50Z; read out 01:30:53 UTC by the pre-declared sequential gate; nn_x/abl/logs/readout_pre2_10ep.txt + .json)

Runs: abl_e10_xdrop_pre2_s951 (GPU N, PID 2490513) and s952 (GPU N, PID 2490879), launched 00:03:48 / 00:04:02 UTC by
nn_x/abl/encdose/launch_e10_xdrop_pre2.sh (0 other compute processes, [n] GB free at launch; the (A) and (C) screens shared the GPUs
later), 85.0 / 84.4 min, rc 0. Logged args == abl_e10_pop1_s951's except variant, xt_add ('' here) and xt_exclude (the 6 pre2 columns
here; verified by parsing both logs). Selected steps 97,500 / 95,000. Reference w5_e10_base_s951 / s952 (nn_x/w5/preds), read directly
(nothing copied into nn_x/w5). Read-out plumbing check: on w5_e10_ap1 vs w5_e10_base the script reproduces w5_eval_E_ap1.txt to every
printed digit with the sign flipped (scratchpad encdose/readout_plumb_ap1.txt).

| WORTH of the pre2 channel at 10 epochs = base - xdrop_pre2 (selected checkpoints, pooled s951 / s952) | value | gate clause |
|---|---|---|
| val | **+0.00077** +- 0.00073 (z +1.1; seeds +0.00070 / +0.00085) | (1) >= +0.0015: **NO -> STOP** |
| holdout | +0.00067 +- 0.00043 (z +1.6; seeds +0.00131 / +0.00004) | (2) > 0: OK (printed, not counted) |
| val M>=50 | +0.00097 (z +1.2); M<50 +0.00049; Tue +0.00066; Thu +0.00091 | (3) > 0: OK (printed, not counted) |
| pmix | +0.00004 +- 0.00110 | information |
| holdout late (Sep 17-19) / early (Sep 6-8) | +0.00100 +- 0.00080 / +0.00075 +- 0.00083 | information (amendment (b)) |
| @last (info) | val +0.00011 +- 0.00073 (seeds +0.00069 / -0.00047), holdout +0.00065 +- 0.00043 | information |
| 2-member pool (info) | val +0.00061 +- 0.00083, holdout +0.00059 +- 0.00049 | information |

**VERDICT (B) by the pre-declared sequential gate: FAIL at clause (1).** The pre2 channel is worth +0.00077 val at 10 epochs (the
2-epoch read gave +0.00128; F9 - F6 gave +0.0009), half the +0.0015 E1 / F8 trigger and 1 SE from zero; the holdout and M>=50 clauses
would have passed but do not count. PRE2-10EP closes: the PRE2-XL re-open condition (GLOBAL_RECAP section 5 row 7) does not fire.
The two runs stay as the pre2-free 10-epoch reference for PSEUDO-DIRECT (SOLVER_POWER 6.3 item 3): checkpoints
nn_x/abl/ckpt/abl_e10_xdrop_pre2_s951/952[@last, @70000, @77500].pt, preds nn_x/abl/preds/nn_abl_e10_xdrop_pre2_s95x[@last]_trainonly_{val,test}.npy
and preds/holdout/.

### 10.E ENCODER-DOSE verdict (2026-11-14T01:31:50Z)

(A) NOT KILLED on val (f 0.63; the holdout analogue would kill), (B) FAIL, (C) NO-GO. By the pre-declared escalation rule (10.D: (C)
must pass AND ((A) >= 40% OR (B) passes)) the encoder sweep is OFF: ENCODER-DOSE closes with 0 GPU-h of escalation. Zero-shot encoder
quality buys a popularity-correlated scalar (the ID part is worth ~+0.0005 val / 0 holdout inside the ranker at 2 epochs), the whole
channel is worth +0.0008 val at 10 epochs, and a frozen 64-d representation of the encoder's states adds nothing (-0.0001 val) to the
6 scalars. GPU spent: (B) 2 x 85 min, (A) 2 x 26 min + 80 s scoring + 2 min smoke, (C) 2 x 27 min + 3 min export + 2 x 2 min smoke +
1 min stale = about 5.6 GPU-h. Nothing fine-tuned, pooled or submitted; reports/ and nn_x/w5/* untouched (the frozen set is unaffected).
