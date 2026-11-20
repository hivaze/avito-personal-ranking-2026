# Neural ranker report (Avito ML Cup 2026, personal ranking)

Status: final (2026-11-11 ~05:45 UTC). All numbers are val full NDCG with the exact `src/metrics.py`.

## TL;DR
* **Best clean train-only single model**: `s2_v2_feed_k9` = target-attention set ranker with raw-cosine similarity bias, listwise production-order distillation (K9), equal per-feed loss weight (K1), domain token + user-0 inputs, checkpoint chosen on a random-like train holdout, then stage 2 on the 164K random-order exploration feeds: **val 0.38271** (no val labels used anywhere).
* **Clean train-only ensemble** (4 v2 members, no val labels): **val 0.3825** (`nn_final2_trainonly_*`).
* **Fine-tuned on val** (5-fold GroupKFold by user_id, zero-init branch with 67 val/test-only features incl. ALS, co-visitation, leave-user-out popularity, near-duplicate context; snapshot mean): single models 0.3852-0.3857 OOF; **final ensemble `nn_final2` OOF: 0.38634** (8 fine-tuned runs over 5 v2 train-stage models).
* Reference: LightGBM val-only CV 0.365-0.367; best GBDT train-fit 0.370.
* What mattered (val, paired where noted): history target attention (+0.006-0.010), similarity bias (+0.0024), production-position distillation (MSE +0.002; listwise K9 a further +0.0024), equal per-feed loss weight in the train stage (+0.0020), fp32 head (+0.001-0.003, removes bf16 ties), stage 2 on exploration feeds (+0.001), snapshot averaging in fine-tune (+0.0017), ALS/covis/near-dup features in the fine-tune branch (+0.0037). Neutral: set self-attention, position tower, width 256, 3 blocks / 8 heads, richer attention bias, domain token + user-0 inputs, category-intent head, history-length TTA, 2 fine-tune seeds, K1 in fine-tune. Negative: BCE (-0.001), up-weighting exploration feeds inside the main run (-0.001).

## Code (`<repo>/nn/`)
| file | purpose |
|---|---|
| `data.py` | `SplitData`: loads a whole split (cand/hist arrays, user emb, int8 item table: train 10.4 GB) into GPU memory; `batches()` buckets feeds by M (padded to x16) into fixed-shape batches (new shapes cost ~0.25 s of kernel selection each, bucketing gave a ~10x speed-up); `gather()` builds padded [B,M]/[B,K] tensors on GPU; `attach_extra()` loads precomputed features for the fine-tune branch |
| `model.py` | `dense_features()` (61 per-candidate features computed on GPU inside each batch), `Ranker`, losses |
| `common.py` | prediction (scatter back to flat cand order), exact full NDCG (`src/metrics.py`), feature-normalization stats; refuses to run unless CUDA_VISIBLE_DEVICES is 6/7 |
| `train.py` | train on the train split, checkpoint by val NDCG (or last step), write train-only val/test predictions; options for ablations, `--sim_bias`, `--aux_pos`, `--prand_base` (exploration-feed weighting), `--only_random` + `--init` (stage 2) |
| `finetune.py` | 5-fold GroupKFold(user_id) fine-tune on val from a checkpoint; `--extra` feature branch; snapshot averaging; OOF val + fold-averaged test predictions |
| `predict.py` | re-predict val/test from a checkpoint |
| `ensemble.py`, `blend.py` | per-feed z-score averaging of variants (fine-tuned and train-only), optional mix of train-only into fine-tuned |
| `stage2_ft.sh`, `mix_ft.sh`, `run_final.sh` | v1 reproduction drivers (`nn_final`) |
| `stage2_ft_v3.sh`, `mix_ft_n2f.sh`, `run_final_v2.sh` | v2 reproduction drivers (`nn_final2`) |
| `logs/`, `ckpt/` | per-run logs (`<variant>.log`, `<variant>_ft.log`, json summaries) and checkpoints |

## Architecture
* **Candidate token** (d=128): `Linear([e_c, e_c * u])` (e_c = L2-normalized item emb, u = normalized user emb) + category embedding + `MLP(61 dense features)` + missing-emb vector, LayerNorm.
* **Dense features (computed on GPU per batch, no precomputed files)**: f1..f5 raw + within-feed z-score + within-feed mid-rank percentile; emb-missing flag, log item norm; cos/dot with user emb; candidate-vs-history cosine: max over all / last 8 / last 32, mean, soft-max pooling (T=0.2), top-3 mean, max per event type (10); category affinity with history: log count, share, log count per event type (10), recency of first same-category event; feed category share; log M; log H; within-feed z and percentile of cos_u, max-sim, soft-sim, top3, cat count. Standardized with train statistics.
* **History tokens**: `Linear(e_h)` + emb(event_type) + emb(event_source) + emb(category, shared) + emb(recency index 0..255) + missing-emb vector; up to 256 newest events; 1 pre-LN transformer layer over [user token, history].
* **Blocks** (x2): target attention (candidates cross-attend over [user, history], 4 heads) then set self-attention over the feed's candidates (no positional encoding; train order is never used as input).
* **Similarity bias (`--sim_bias`)**: the raw cosine between the candidate item embedding and each history item embedding (user embedding for the user token) is added to the target-attention logits with a learned per-head scale (init linspace(0, 8)), so some heads do exact nearest-history-item retrieval in the original embedding space.
* **Head**: LN -> Linear -> GELU -> Linear -> logit, plus a linear "wide" term on the standardized dense features; **computed in fp32** (see precision note).
* **Auxiliary production-position head (`--aux_pos 0.5`)**: a second head regresses pos/(M-1) (MSE) on train feeds = distillation of the production ranker, whose order is informative but unavailable at test. Weighted by 1-p_random (random-order feeds carry no production order).
* **Position tower (ablation)**: additive train-only logit `g(pos) + g'(M-1-pos)` (scalar embeddings), dropped at inference. It learns a top-of-feed effect (+1.1 logit at pos 0, smooth decay) and a strong end-of-feed effect (+0.6..0.8 for the last 3 positions vs -0.7 elsewhere, i.e. the feed is logged until the user stops, often right after a click).
* **Loss**: listwise softmax cross-entropy per feed with target distribution y/sum(y).
* Optimizer AdamW (lr 1e-3, wd 0.02, betas 0.9/0.98), warmup 500 + cosine to 3%, bf16 autocast (fp32 head), grad clip 1.0; batches of <=512 feeds / <=32768 candidates; 2 epochs = ~20K steps. 0.86M parameters.
* **Speed**: the whole train split lives on one GPU (~21 GB); 2 epochs over 3.15M feeds take 12-20 min wall-clock while sharing the GPU with 2-3 other jobs (~25-30 steps/s alone).

### Precision note (important)
The first runs produced bf16 logits. bf16 quantizes scores (step 1/64 for |x| in [2,4)), creating within-feed ties that the metric breaks pessimistically: e.g. `nopos` 0.37218 (bf16) vs 0.37408 with random tie-breaking. The head now runs in fp32 and all train-only checkpoints were re-predicted (same weights). Fine-tuned snapshot predictions are averages and have no ties.

## Results (val full NDCG, 14,423 feeds; SE of a single score ~0.0019; paired differences between models are much less noisy; seed-to-seed spread of the same config ~0.0005-0.001)
Reference: random 0.290, best single feature 0.349, LightGBM lambdarank val-only 5-fold OOF 0.365-0.367.

### Train-stage ablations (all 3.15M train feeds, 2 epochs, checkpoint = best of evals every 1000 steps)
| variant | change | params | val (bf16 head, as logged) | **val (fp32 head)** | time |
|---|---|---|---|---|---|
| base | hist attn + set attn + pos tower, softmax | 0.84M | 0.37047 | 0.37361 | 18.4 min* |
| nopos | base without position tower | 0.84M | 0.37220 | 0.37385 | 17.8 min* |
| nohist | base without history tokens / target attention (dense hist-sim features kept); peaks at 2k steps, then degrades | 0.65M | 0.36642 | 0.36808 | 11.3 min |
| noset | base without set self-attention | 0.58M | 0.37032 | 0.37386 | 13.0 min |
| bce | base with pointwise BCE instead of listwise softmax | 0.84M | 0.36928 | 0.37272 | 14.9 min |
| big_nopos | nopos, d=256, 8 heads | 3.0M | 0.37144 | 0.37271 | 20.0 min |
| nopos_aux | nopos + aux production-position head (w=0.5) | 0.86M | 0.37399 | 0.37581 | 12.4 min |
| rev_aux | aux + end-of-feed-only position tower g'(M-1-pos) | 0.86M | 0.37371 | 0.37483 | 22.2 min |
| nopos_sim | nopos + similarity bias | 0.84M | 0.37524 | 0.37624 | 20.9 min |
| **nopos_aux_sim** | nopos + aux + sim bias (seed 2) | 0.86M | 0.37735 | **0.37848** | 20.7 min |
| aux_sim_s4 / s5 / s9 | same, seeds 4 / 5 / 9 | 0.86M | 0.37744 / 0.37732 / - | 0.37863 / 0.37814 / 0.37805 | 18-26 min |
| aux_sim_b3 | 3 blocks (seed 7) | 1.2M | 0.37703 | 0.37806 | 15.5 min |
| aux_sim_h8 | 8 heads (seed 6) | 0.86M | 0.37651 | 0.37742 | 26.4 min |
| aux_sim2 | richer attention bias (per event type sim scale/prior, same-category, recency); stopped at 11k steps | 0.86M | 0.37431 (10k) | within seed noise of aux_sim at equal steps | - |
| nopos_e4 | nopos, 4 epochs; stopped at 24k/40k steps | 0.84M | 0.37257 (20k) | - | - |
| nopos_aux1 | aux weight 1.0; stopped at 6k steps, tracked w=0.5 | 0.86M | 0.36797 (5k) | - | - |
| pw_aux_sim | aux_sim with per-feed loss weight 0.2 + p_random (exploration feeds up-weighted ~6x; aux loss weighted 1-p) | 0.86M | - | 0.37704 (worse than the 4 unweighted seeds 0.3780-0.3786) | 11.5 min |

(*) before shape bucketing. Times are wall-clock with the GPU shared by 2-4 jobs.

Takeaways: history target attention matters most; with exact (fp32) scores the position tower and set attention are neutral (base 0.3736 vs nopos 0.3739 vs noset 0.3739); listwise > BCE; width does not help; the similarity bias (+0.0024 over nopos) and the production-position distillation (+0.002) add up (+0.0046).

### Stage 2: continue training on random-order exploration feeds (train feeds with M>=50 and p_random>0.5, 164K feeds; from the coordinator's detector)
`train.py --init <ckpt> --only_random 0.5 --lr 2e-4 --epochs 2 --keep last` (no val labels used; last step kept).
| start | train-only before | after stage 2 |
|---|---|---|
| nopos_aux_sim (first try, 3 epochs, evals every 200 steps) | 0.37848 | 0.37996 (last), best eval 0.38004 |
| nopos_aux_sim (final recipe: 2 epochs, last step) | 0.37848 | 0.37988 |
| aux_sim_s4 | 0.37863 | 0.37905 |
| aux_sim_s5 | 0.37814 | 0.37931 |
| aux_sim_b3 | 0.37806 | 0.37941 |
| aux_sim_s9 | 0.37805 | 0.37966 |

Up-weighting the exploration feeds inside the main training run (pw_aux_sim) hurt slightly, but a short second stage on them after normal training helps consistently (+0.0004 to +0.0016 train-only, mean +0.0012).

### Val fine-tune (5-fold GroupKFold over val feeds by user_id, same folds as `src/cv_val.py`; test = mean of the 5 fold models)
| start checkpoint / config | train-only val | fine-tuned OOF |
|---|---|---|
| base, lr 1e-4, best epoch | 0.37046 (bf16) | 0.37195 |
| nopos, lr 1e-4, best epoch | 0.37218 (bf16) | 0.37322 |
| nopos + extra item features (29) via zero-init branch, branch lr x10, best epoch | 0.37218 | 0.37490 |
| nopos + extra item features, branch lr x3, best epoch | 0.37218 | 0.37505 |
| nopos + extra item + time-aware target encodings (41 features), best epoch | 0.37218 | 0.37534 |
| nopos + extra item_safe (27 features), best epoch | 0.37218 | 0.37530 |
| **nopos + extra item_safe, snapshot mean of epochs 1..4** | 0.37218 | **0.37704** |
| nopos + item_safe, branch lr x3, snapshot of epochs 1..6 | 0.37218 | 0.37714 |
| nopos_aux + item_safe, snapshot 1..4 | 0.37581 | 0.37819 |
| nopos_sim + item_safe, snapshot 1..4 | 0.37624 | 0.37900 |
| nopos_aux_sim + item_safe, lr 1e-4, snapshot 1..4 | 0.37848 | 0.37995 |
| nopos_aux_sim + item_safe, lr 1e-4, 2 fine-tune seeds per fold, snapshot 1..4 | 0.37848 | 0.38003 |
| **nopos_aux_sim + item_safe, lr 5e-5, snapshot 1..6** | 0.37848 | **0.38068** |
| aux_sim_s5 / s4 / b3 / s9 + item_safe, lr 1e-4, snapshot 1..4 | 0.3781 / 0.3786 / 0.3781 / 0.3781 | 0.38069 / 0.37964 / 0.37964 / 0.37924 |
| s2_nopos_aux_sim (stage 2) + item_safe, lr 5e-5, snapshot 1..6 | 0.37988 | 0.38097 |
| s2_nopos_aux_sim + item_safe + 12K random-order train feeds mixed into every fine-tune epoch (`--add_random 12000`) | 0.37988 | 0.38139 |
| s2_aux_sim_s5 / s2_aux_sim_b3 (stage 2) + item_safe, lr 5e-5, snapshot 1..6 | 0.37931 / 0.37941 | 0.38076 / 0.38022 |
| val-only from scratch (same arch; stopped after 2 folds) | - | fold0 ~0.352, fold1 ~0.366 (fine-tuned from train: 0.365 / 0.372) |

Fine-tune details: AdamW lr 1e-4 (5e-5 final), wd 0.02, 30 warmup steps, batches of <=128 feeds / 8192 candidates, position tower off, listwise loss. Extra branch: `item_safe` = within-split candidate-count rank/z, train-period item impressions, transductive item popularity in all histories (+rank/z/gap, per event type), train item CTR/COEC (+rank/z/gap), item train positives, user train CTR / #feeds, hour, centrality; log1p for counts, standardized on val; raw within-split candidate counts excluded because their scale differs between val (3 days) and test (4 days). The single-epoch curves are noisy (best epoch 1-4 varies by fold); averaging the per-feed z-scored predictions of all epochs (snapshot ensemble) is +0.0015-0.002 over the best single epoch and needs no epoch selection.

### Ensembles (per-feed z-score mean)
| members | train-only ensemble | fine-tuned ensemble | fine-tuned mixed with 0.2 x train-only (= snapshot over epochs 0..E) |
|---|---|---|---|
| stage 1: nopos_aux_sim, aux_sim_s5, s4, b3, s9 (`nn_ens_stage1_*`) | 0.37900 | 0.38082 | 0.38108 |
| **final v1**: 8 stage-2 models (nopos_aux_sim, aux_sim_s4/s5/s9/s11/s12, b3, h8), each fine-tuned twice (plain + `--add_random 12000`) = 16 fine-tuned runs (`nn_final_*`) | 0.38064 | 0.38133 | 0.38184 |
| **final v2 (`nn_final2_*`)**: 8 n2full fine-tunes of 5 v2 models; train-only = 4 v2 models (s2_v2_feed, s2_v2_k9, s2_v2_idcg, s2_v2_feed_k9) | **0.38246** | **0.38634** | not used |

Mixing 12K exploration train feeds into each fine-tune epoch: v1 (item_safe) mean over 8 pairs +0.0005 (7 of 8 positive); v2 (n2full) +0.0003 / +0.0001 / -0.0009 on 3 pairs, i.e. neutral. Adding the v1 ensemble to the v2 one: +0.00003 at weight 0.1, negative beyond. The 7-run version without the intent member scores 0.38656 (within noise); all v2 runs were kept for diversity rather than selected on val.

## Deliverables (`<repo>/preds/`, float32, aligned with `data/flat/{val,test}/cand_*.npy`)
| file | what | val NDCG |
|---|---|---|
| **`nn_final2_val_oof.npy` / `nn_final2_test.npy`** | **final**: per-feed z-mean of 8 v2 fine-tuned runs, n2full branch: s2_v2_feed (plain 0.38535, mix 0.38567), s2_v2_k9 (0.38555, mix 0.38567), s2_v2_idcg (0.38517), s2_v2_feed_k9 (0.38572, mix 0.38478), s2_v2_intent (0.38545); no train-only mixing (w=0.2 no longer helps with n2full: 0.38597 vs 0.38605 on 4 members) | **0.38634** (OOF) |
| **`nn_final2_trainonly_val.npy` / `nn_final2_trainonly_test.npy`** | clean train-only ensemble: s2_v2_feed, s2_v2_k9, s2_v2_idcg, s2_v2_feed_k9 (no val labels; checkpoints chosen on the train holdout) | 0.38246 |
| `nn_s2_v2_feed_k9_trainonly_{val,test}.npy` | best clean train-only single model | 0.38271 |
| `nn_s2_v2_feed_k9_n2f_{val_oof,test}.npy` | its n2full fine-tune | 0.38572 |
| `nn_final_{val_oof,test,trainonly_val,trainonly_test}.npy` | previous final (v1: 8 stage-2 models x 2 item_safe fine-tunes, train-only mixed in at 0.2) | 0.38184 OOF / 0.38064 train-only |
| `nn_ens_stage1_*` | stage-1 ensemble (5 models) | 0.38108 / 0.37900 |
| `nn_<variant>_trainonly_{val,test}.npy` | train-only predictions of every checkpoint (e.g. base, nopos, nohist, noset, bce, nopos_aux_sim, aux_sim_s*, s2_*, s2_v2_*) | see tables |
| `nn_<variant>_{val_oof,test}.npy` | fine-tuned OOF / fold-averaged test of every fine-tune run | see tables |
Note: v1 stage-1 checkpoints were selected on val (mildly val-informed); v2 checkpoints are selected on the random-like train holdout and stage 2 keeps the last step, so v2 train-only predictions use no val labels. Fine-tune folds = `GroupKFold(5)` over val feeds in `feeds.parquet` order with groups=user_id (same as `src/cv_val.py`).

### Playbook items K1 / K10 (coordinator's list)
* **K1 loss normalisation, fine-tune A/B** (s2_nopos_aux_sim, same folds/seed, lr 5e-5, snapshot 1..6): (a) per-batch mean (current) 0.38097; (b) equal per-feed weight, sum/max_feeds 0.38088; (c) equal x K/IDCG(K) 0.38113. All within ±0.0002 (paired SE ~0.0005): neutral in fine-tune (val batches are small and M varies less).
* **K1 in the train stage**: see v2 runs below (equal per-feed weight looks clearly better at equal steps).
* **K10 TTA over history length** (train-only, val): max_hist 64 / 128 / 192 / 256 = 0.3754 / 0.3782 / 0.3794 / 0.3799 for s2_nopos_aux_sim; every z-mean combination is worse than 256 alone (best (192,256) 0.37975). Rejected.

### v2 train stage: K1 + K2 + K5 + random-like holdout (item 4) + K9
All v2 runs: nopos + aux + sim_bias, plus `--user0` (K5: is_user0, is_user0*log(1+H) dense inputs), `--domain` (K2: feed-level domain embedding added to candidate tokens and head input; train feeds = production soft-mixed toward random-like by p_random for M>=50; train-only val/test inference with random-like; fine-tune switches val/test to an exploration embedding initialised from random-like), and `--holdout 30000` (fixed 30K random-like train feeds with p_random>0.9, M>=50 excluded from training; **checkpoints are now selected on this holdout, so val numbers are clean**). Holdout NDCG is lower in level (all M>=50).
| run (seed 21 unless noted) | difference | train-only val | holdout | stage 2 val / holdout | fine-tune OOF (plain / mix) |
|---|---|---|---|---|---|
| v2_batch | K1 (a) per-batch mean | 0.37804 | 0.29619 | - | - |
| v2_feed | K1 (b) equal per-feed weight | 0.38003 | 0.29676 | 0.38067 / 0.29755 | item_safe 0.38162 (mix 0.38182); n2full 0.38535 |
| v2_idcg | K1 (c) equal x K/IDCG(K) | 0.38004 | 0.29749 | 0.38054 / 0.29843 | n2full 0.38517 |
| v2_k9 | K1 (a) + K9 listwise production-order distillation (CE toward softmax(-pos/10), w 0.5) | 0.38039 | 0.29892 | 0.38123 / 0.29964 | n2full 0.38555 / mix 0.38567 |
| **v2_feed_k9** (seed 23) | K1 (b) + K9 | **0.38176** | **0.30091** | **0.38271 / 0.30094** | n2full 0.38572 / mix 0.38478 |
| v2_intent | v2_feed + N3 category-intent aux head (w 0.1) | 0.37866 | 0.29702 | 0.37939 / 0.29750 | n2full 0.38545 |

* K9 (listwise production-order distillation instead of MSE): +0.0024 val / +0.0027 holdout over v2_batch (paired, same seed) - the largest train-stage gain of the v2 round. K1(b) + K9 combined (v2_feed_k9) is the best train-only model: 0.38176 val / 0.30091 holdout, 0.38271 / 0.30094 after stage 2.
* N3 category-intent head (paired with v2_feed): -0.0014 val, +0.0003 holdout: not accepted (it may need a stronger target, e.g. weight 0.3 or a stop-gradient).
* Fine-tune with n2full shrinks the gap between members: all v2 members land at 0.3852-0.3857.

* K9 (listwise production-order distillation instead of MSE): +0.0024 val / +0.0027 holdout over v2_batch (paired, same seed) - the largest train-stage gain of the v2 round.

* K1 in the train stage: equal per-feed weight beats the per-batch mean by +0.0020 val / +0.0006 holdout (paired, same seed). In the train stage the per-batch mean gives an M~300 feed ~4.7x the weight of an M<=64 feed (512 vs ~109 feeds per batch), much more skewed than in fine-tune, which explains why K1 is neutral in fine-tune but helps here.
* K2 + K5 (v2_batch vs the unchanged stage-1 seeds 0.3780-0.3786): neutral in train-only.

### N1 / N2: richer val/test-only features in the fine-tune branch (start: s2_v2_feed, train-only 0.38068; lr 5e-5, snapshot 1..6)
| extra branch | features | fine-tuned OOF |
|---|---|---|
| item_safe (previous default) | 27 | 0.38162 |
| item_luo (N1: it_hist_* replaced by leave-same-user-out popularity) | 27 | 0.38123 (the old version was ~0.0004 optimistic) |
| n2 = item_luo + ALS item-item profiles (9) + co-visitation (9) | 45 | 0.38437 |
| **n2full** = n2 + sequence-similarity (seq_max_all, seq_top3, seq_dcos_hl5, seq_nd90_cnt, seq_argmax_hpos) + near-duplicate context ctx_nd_* (17) | 67 | **0.38535** |
The collaborative (ALS, co-visitation) and near-duplicate-context signals are not available to the train-stage NN and add +0.0037 over item_safe on the same checkpoint. All v2 members are fine-tuned with n2full.

### Complementarity with the team's GBDT predictions (simple per-feed z-score blends on val, NN = preliminary `nn_final` 0.38186)
GBDT/stack predictions in `preds/` score 0.364-0.370 alone and correlate 0.85-0.875 with the NN; blends w*NN + (1-w)*GBDT give at most 0.3820 (w=0.8, xgb_rand50) vs 0.38186 for the NN alone, i.e. the GBDTs add almost nothing at the simple-blend level. The NN train-only predictions are the cleanest stacking inputs (no val labels used).

## Notes / ideas not pursued
* Cross-feed "later history" leak (candidate appears in a later history of the same user, ~0.8% of val feeds, ~93% positive) is handled by the coordinator as post-processing.
* The tte_* time-aware target encodings were weak in the fine-tune branch (+0.0003) and not used in the train stage.
* Not done for lack of time (coordinator list): N4 item-ID residual with ID-dropout, N5 typed attention bias + SiLU pooling channel (a first typed-bias variant `aux_sim2` was neutral at 10k steps), N6 test pseudo-labels, K3 val-teacher distillation, K4 event-type aux head; a train-stage version of the ALS/co-visitation signals (they only exist for val/test today) is the most promising next step given the +0.0037 they give in fine-tune.
* Ensemble breadth: v2 members differ in loss normalisation / distillation type / seed; seeds alone saturate (16 v1 runs = +0.0005 over the best single).

---------------------------------------------------------------------------------------------------------------------

# Round 2 (2026-11-11 05:33-09:15 UTC) - DONE; final numbers in "Round 2 final deliverables" at the end
Baseline for paired comparisons: `v2_feed_k9` (seed 23): train-only val 0.38176 / holdout 0.30091; stage 2 0.38271 / 0.30094; n2full fine-tune 0.38572.

New code:
* `--xt cv` (train.py): 16 leak-safe train-stage collaborative inputs appended to the dense features: co-visitation `seq_cv_{invd20,w5,fwd5,t1,seqs,pop,norm,invd20_rk,invd20_z}` (train rows of day d use only feeds of days < d) and leave-same-user-out popularity `it_hist_{all,et1..et4}_luo` (+ all _rk/_z); fixed transform (log1p for counts, _z clipped), stored fp16 on GPU (7.9 GB for train).
* `--id_buckets 4194304` (N4): hashed item-ID table (2^22 x 32, zero-init so cold items == dropped IDs, ID-dropout 0.5 per token in training) projected into both candidate and history tokens; frozen and dropout-free in fine-tune.
* `finetune.py --pseudo <teacher>` (N6): per-fold test predictions are now saved (`nn_<run>_test_folds.npy`); a student fine-tune on fold k mixes 12K test feeds per epoch with soft targets softmax(z_teacher_k / T), weight 0.3, where teacher_k is the fold-k model of the teacher run (never saw val fold k).
* `--arch gru` (bi-GRU history encoder instead of the transformer layer) and `--arch din` (DIN-style 2-head additive MLP attention with cosine bias + SiLU pooling channel + DCNv2 cross network; no set attention, no history self-attention).

## Round 2 results (interim, updated as runs finish)
### Feature audit after the "later history" finding
* Train-stage NN inputs use only the feed's own history/candidates (+ leak-safe `seq_cv_*`/`*_luo` in v3): clean.
* Fine-tune branch: `item_safe` (used in `nn_final`, v1) contains non-LUO `it_hist_*` (include the same user's later snapshots) -> **v1 `nn_final` is contaminated**; do not use it. `n2full` (used in `nn_final2`) already had LUO popularity but also contains (a) in-split candidate counts `it_cand_cnt_split_rk/_z` (the item's candidate appearances include the same user's later feeds) and (b) `als_*`, whose item factors were trained on train+val+test histories including the same users' later snapshots (for rare items a factor can be dominated by that user's later history).
* New audited preset **`n2clean`** = train-period item/user stats, hour, centrality, LUO popularity, causal co-visitation (same-user sequences skipped), own-history sequence similarity, near-duplicate context; no ALS, no in-split counts.
* Measured on s2_v2_feed (train-only 0.38068): n2full 0.38535, n2clean+ALS 0.38544 (in-split counts worth ~0), **n2clean 0.38471** (ALS worth +0.0007, possibly partly leak). All round-2 fine-tunes use n2clean.

### Train-stage collaborative inputs (`--xt cv`) - clear win
Paired with v2_feed_k9 (same config and seed 23):
| run | train-only val | random-like holdout |
|---|---|---|
| v2_feed_k9 (baseline) | 0.38176 | 0.30091 |
| **v3_cv** (+16 leak-safe co-visitation / LUO-popularity inputs) | **0.38491 (+0.0032)** | **0.31011 (+0.0092)** |
| v3_id (+ hashed item-ID residual, N4) | 0.37953 (-0.0022) | 0.29734 (-0.0036) |
The gain is visible from the first 1000 steps (+0.006 val / +0.007 holdout) and stays through training. The larger holdout gain is expected: train-distribution items are well covered by the co-visitation source, while val/test have item churn (co-visitation non-zero for 21% of train candidates vs 11% val / 10% test).

### N4 item-ID residual - negative
Hashed 2^22 x 32 table, zero-init, ID-dropout 0.5, in candidate and history tokens. The holdout-selected checkpoint (step 10000) is -0.0022 +- 0.0008 val overall; in the 2nd epoch the model collapses (val 0.372, memorisation of train item IDs). Split by ID coverage of the feed's positives (71% of val candidates and 77% of val positives have an item id seen in train; test 61%): feeds whose positives are all covered -0.0002 +- 0.0010 (neutral), feeds with no covered positive **-0.0075 +- 0.0021**: the ID table teaches the model to prefer known items, which hurts cold positives. Pipeline stopped; not used.

### N6 test pseudo-labels - no gain
Teacher = n2full fine-tune of s2_v2_feed_k9 re-run with per-fold test predictions saved (OOF 0.38580). Student = same fine-tune where every epoch also trains on 12K test feeds with soft targets softmax(z_teacher_fold_k), weight 0.3 (fold k's teacher never saw val fold k): OOF **0.38535 (-0.0004)**. Not pursued further.

### Feature files deleted under us (05:52)
`seq_cv_w5`, `seq_cv_fwd5`, `seq_cv_t1`, `seq_cv_seqs`, `seq_cv_invd20_z` (and `seq_dcos_hl5`, `seq_nd90_cnt` for val/test) were pruned from `data/feats/*` by the feature workflow (`feats_v2/seq_/KEEP.txt`) while v3 runs were using them. v3_cv (16 inputs) keeps its train-only val/test predictions, but its stage 2 / fine-tune could not run. All presets now drop missing files at load time (and log it); the co-visitation members were retrained as **v4** with the 11 remaining inputs (`seq_cv_{invd20,pop,norm,invd20_rk}` + LUO popularity), and `n2clean` now has 49 features.

### Leak-free ALS (from the coordinator, f=64, own val/test candidate pairs removed)
The old `als_*` were trained on pairs from the users' later snapshots (+0.0005 val inflation, concentrated on the 117 feeds whose user has a later feed). All round-2 fine-tunes use preset **`n2clean_als`** = n2clean + the new leak-free ALS (49 + 9 = 58 features). Re-fine-tuned v2 members with it: s2_v2_feed_k9 0.38547 (was 0.38572 with n2full), s2_v2_k9 0.38466 (was 0.38555). `nn_final2` (built on the leaky ALS) is therefore ~0.0005-0.001 optimistic on val.

### v4 / v5 members (11 co-visitation/LUO inputs; checkpoints chosen on the random-like holdout; pipeline = stage 2 + n2clean_als fine-tune)
| member | arch / difference | train-only val / holdout | stage 2 val / holdout | fine-tuned OOF |
|---|---|---|---|---|
| v4_cv (seed 23) | tf, K1 feed | 0.38469 / 0.30999 | 0.38532 / 0.30986 | 0.38636 |
| v4_cv_s33 | tf, K1 idcg | 0.38474 / 0.30956 | 0.38593 / 0.30981 | 0.38633 |
| v4_cv_din (seed 32) | DIN MLP attention + SiLU pooling + DCNv2, no set attn, listwise+BCE | 0.38263 / 0.30753 | 0.38366 / 0.30780 | 0.38453 |
| v5_cvix (seed 23) | v4_cv + 11 item cross-feed context inputs (ctx_ix_*: feed-size stats, production-position prior, f1 prior; day cutoff) | 0.38565 / 0.31019 | TBD | TBD |
v5 vs v4_cv (paired): +0.0010 val but only +0.0002 holdout -> not a clear win by the holdout rule; kept as an ensemble member.

Marginal value of architecture diversity (val, equal-weight z-means): tf x2 fine-tunes 0.38604; + din 0.38675 (+0.0007 although din alone is 0.38453); train-only tf x2 0.38570 -> + din 0.38611. Blending with the older non-co-visitation generation also helps: z-mean(nn_final2, tf x2) 0.38729, z-mean(nn_final2, din) 0.38734 (nn_final2 itself 0.38634, slightly optimistic).

### v4_cv_gru and preliminary nn_final3 (06:40)
* v4_cv_gru (bi-GRU history encoder, seed 31): train-only 0.38469 / holdout 0.30987, stage 2 0.38543 / 0.30994, fine-tune **0.38664** (best single fine-tune; GRU training is ~1.8x slower).
* `ctx_ix_*` files were also deleted from data/feats (06:20); v5_cvix trained but cannot predict test and is dropped. To stop this from breaking models again, every precomputed feature a model uses is now read through hard links in `nn/feat_cache/<split>/` (`data.feat_path`), created on first use (no extra disk). Caveat: a feature that is later *rewritten* upstream keeps its old content in the cache (delete the cache entry to refresh).
* **Preliminary `nn_final3`** (equal-weight per-feed z-means, members pre-declared: all v4 co-visitation members + the two strongest v2 members for diversity):
  * fine-tuned (n2clean_als): s2_v4_cv, s2_v4_cv_s33, s2_v4_cv_din, s2_v4_cv_gru, s2_v2_feed_k9, s2_v2_k9 -> **0.38737** OOF (clean; nn_final2 0.38634 was ALS-inflated)
  * **clean train-only** (`nn_final3_trainonly_*`, same 6 stage-2 models, no val labels): **0.38689** (nn_final2_trainonly 0.38246, +0.0044)

### More round-2 members / negatives (06:47)
* v6_cvhn = v4_cv + 10 candidate-competition inputs (`ctx_hn_*`, `ctx_hs_*`: reciprocal-NN and soft attention mass of the feed's candidates over the user's recent history items): 0.38508 / holdout 0.31093 vs v4_cv 0.38469 / 0.30999 (paired, +0.0004 / +0.0009): small, kept as a member.
* v4_cv_din_s34 (second DIN seed): 0.38232 train-only.
* v4_cv_hid = v4_cv + item-ID embeddings in **history tokens only** (no candidate-ID shortcut): on par with v4_cv up to 10k steps (0.38302 vs 0.38328), then the same 2nd-epoch collapse as the full ID residual (12k: 0.37961 vs 0.38409). Negative; stopped.
* v6_cvhn stage 2 0.38581 / 0.31117 -> fine-tune 0.38530; v4_cv_din_s34 stage 2 0.38357 -> fine-tune 0.38287: for these two the fine-tune (snapshot of epochs 1..6) is slightly *below* the train-only stage-2 model; per-epoch curves peak at epoch 1 and decay. With co-visitation inside the train stage the val fine-tune adds less (+0.0004..+0.0012 for the other v4 members) and overfits faster. Note also that 7 more `ctx_nd_*` files (c80/c90/c95, dupb_*, mean) were deleted upstream between fine-tunes, so later fine-tunes use 51 instead of 58 branch features (each run is internally consistent for val and test).
* v4_cv_gru_s35 (second GRU seed): train-only 0.38570 / holdout 0.30995, stage 2 0.38643 / 0.30986, fine-tune **0.38704**.
* Interim 9-member ensemble (v4_cv, v4_cv_s33, gru x2, din x2, v6_cvhn, v2_feed_k9, v2_k9): fine-tuned 0.38740; train-only 0.38651; equal-weight z-mean of the fine-tuned and train-only ensembles (w_trainonly 0.5) **0.38787** (w 0.2: 0.38753). Since the fine-tune now adds little and overfits fast, the final uses the equal-weight mix (one of three coarse values {0, 0.2, 0.5}).

### R2 step 1: fine-tune branch ablation by group (s2_v4_cv, 5 folds, same seeds, current feature files)
Cold-positive feeds = all positives have an item id never seen in train candidates/histories (17.4% of val); warm = all positives seen (70.9%). Deltas vs the train-only stage-2 model (0.38532; cold 0.34662, warm 0.39196):
| fine-tune branch | all | cold-positive | warm-positive |
|---|---|---|---|
| zeroed branch (backbone re-calibration only) | -0.00000 +- 0.00063 | **+0.00986 +- 0.00174** | **-0.00320 +- 0.00075** |
| full n2clean_als (51 features now) | +0.00001 +- 0.00081 | +0.01194 +- 0.00193 | -0.00376 +- 0.00099 |
| without ALS | +0.00008 | +0.01311 | -0.00407 |
| without neighbour context (ctx_nd_* + centrality) | -0.00079 | +0.00915 | -0.00421 |
| without train-period item/user stats | +0.00002 | +0.01090 | -0.00340 |
| without LUO + co-visitation | -0.00006 | +0.01180 | -0.00394 |
Relative to the full branch, only removing the neighbour group hurts (-0.0008 all, -0.0028 +- 0.0011 cold); ALS, train stats and LUO/co-visitation are neutral (clean ALS: removing it is +0.0001). Conclusions: (1) most of the fine-tune's cold-side gain is plain re-calibration of the backbone on val labels (val/test are colder than train), paid for on warm feeds; (2) the only branch group that adds information is the near-duplicate/neighbour context; (3) this fine-tune now gives ~0 overall for s2_v4_cv (0.38533) vs 0.38636 when 7 more `ctx_nd_*` files still existed - the deleted near-duplicate features (c80/c90/c95, dupb_*, mean) carried that +0.001.

### R2 step 2 (running)
* `--xt cvraw --xt_drop q`: the co-visitation/LUO inputs as raw counts, their within-feed rank/z computed inside the model *after* an ID-input dropout that blanks the ID-based inputs of a fraction q of train candidates (cold simulation; test has more zeros than val, train fewer). Sweep q in {0, 0.15, 0.3} (v7_q*, seed 23), chosen on the holdout.
* `--xt cvnd`: cvraw + the train-available neighbour/near-duplicate context (`ctx_nd_{hcent,knn_*,max,top5h,top5u,ucent}`) + candidate competition (`ctx_hn_*`, `ctx_hs_*`), ID dropout only on the ID-based columns (v8_nd_q15).

### R2 step 2: ID-input dropout q (train-only, seed 23, `--xt cvraw`, holdout-selected)
| run | val | holdout | cold-positive feeds vs q0 | warm-positive feeds vs q0 |
|---|---|---|---|---|
| v7_q0 | **0.38578** | **0.31002** | - | - |
| v7_q15 | 0.38538 (-0.0004 +- 0.0006) | 0.30971 | +0.0129 +- 0.0017 | -0.0043 +- 0.0008 |
| v7_q30 | 0.38506 (-0.0007 +- 0.0007) | 0.30908 | +0.0130 +- 0.0019 | -0.0047 +- 0.0008 |
| v4_cv (precomputed rank/z inputs) | 0.38469 (-0.0011 +- 0.0006) | 0.30999 | +0.0029 | -0.0026 |
ID-input dropout reproduces exactly the fine-tune's re-calibration (big cold gain, warm loss) inside the train stage, but the holdout prefers q = 0 and warm feeds lose more than 2 SE; at the test cold/warm mix (22% / ~66%) the net is ~0. Rejected by the acceptance rule (q = 0 chosen). Computing the within-feed rank/z of the raw co-visitation/LUO counts inside the model (v7_q0) is +0.0011 val over the precomputed versions (v4_cv), same holdout.

### Pretrained sequence-encoder features in the fine-tune branch (pre_*, val/test only)
On s2_v4_cv, paired: n2clean (no ALS) 0.38548 vs n2clean_als 0.38533 (+0.0001); **n2clean + pre_* (`n2c_pre`, 60 features) 0.38695 (+0.0015 +- 0.0005 over n2clean**; warm +0.0017, cold -0.0001); n2clean_als + pre_* 0.38612. All final members are re-fine-tuned with `n2c_pre` (suffix `_n2cp`).

### Neighbour / near-duplicate context in the train stage (`--xt cvnd`)
v8_nd_q15 vs v7_q15 (paired: same seed and q = 0.15, only the 20 neighbour/competition inputs added): val 0.38536 vs 0.38538 (0.0000), **holdout 0.31107 vs 0.30971 (+0.0014)** - the highest holdout of any run. A q = 0 version (v8_nd_q0) is running as a final member.
Other members finished: v6_cvhn_gru (GRU + competition inputs) train-only 0.38568 / holdout 0.31027, stage 2 0.38645; v4_cv_gru_s37 0.38484 / 0.30984, stage 2 0.38613; v4_cv_s38 0.38469 / 0.30986, stage 2 0.38596.
Re-fine-tunes with `n2c_pre` so far: s2_v4_cv 0.38695, s2_v4_cv_s33 0.38712, s2_v4_cv_s38 0.38650, **s2_v4_cv_gru 0.38781**, s2_v6_cvhn 0.38672, s2_v6_cvhn_gru 0.38640, s2_v4_cv_din_s34 0.38361 (each +0.0005..+0.0012 over its n2clean_als fine-tune).

### Member selection on the train holdout (no val labels; `nn/holdout_eval.py`, preds in `preds/holdout/`)
Single stage-2 members on the 30K random-like holdout: tf 0.3098-0.3101, GRU 0.3099-0.3103, DIN 0.3075-0.3078, v6_cvhn 0.3112, v6_cvhn_gru 0.3104, v2 members 0.2996-0.3009. Equal-weight z-mean ensembles: all 13 members 0.31105; without v2 0.31102; without DIN 0.31098; without both 0.31117 (all within +-0.0002); tf-only 0.31056 (-0.0005 +- 0.0003), GRU-only 0.31084, tf+GRU 0.31064. Architecture diversity helps on the holdout too; dropping the weaker members does not. Final rule: all members, equal weights.

### v8_nd_q0 and nn_final3 (08:00)
* **v8_nd_q0** (`--xt cvnd`, q = 0, seed 23): train-only 0.38625 / **holdout 0.31168** (paired vs v7_q0: +0.0005 val, +0.0017 holdout; with q = 0.15: +0.0000 / +0.0014), stage 2 0.38643 / 0.31175, n2c_pre fine-tune 0.38763. The near-duplicate/neighbour context is accepted for the train stage (holdout gain in both paired comparisons).
* **`nn_final3`** = 15 members, all pre-declared, equal weights: tf x4 (v4_cv, v4_cv_s33, v4_cv_s38, v7_q0), tf+neighbour x2 (v8_nd_q0, v8_nd_q15), GRU x3, DIN x2, competition inputs x2 (v6_cvhn, v6_cvhn_gru), v2 x2 (v2_feed_k9, v2_k9); each = stage-2 model + `n2c_pre` fine-tune.
  | | val | cold-positive (17.4%) | warm-positive (70.9%) |
  |---|---|---|---|
  | `nn_final3_trainonly` (clean, no val labels) | **0.38704** | 0.35163 | 0.39273 |
  | fine-tuned ensemble alone | 0.38813 | | |
  | **`nn_final3`** (fine-tuned + 0.2 x train-only, the round-1 default = snapshot over fine-tune epochs 0..E) | **0.38828** | 0.35950 | 0.39168 |
  | w_trainonly 0 / 0.5 | 0.38813 / 0.38800 | | |
  * vs nn_final2_trainonly: +0.0046 +- 0.0010 (cold -0.0214 +- 0.0023, warm +0.0121 +- 0.0013): co-visitation inputs act as a warm-item prior.
  * fine-tune vs train-only: +0.0012 +- 0.0007 (cold +0.0079 +- 0.0016, warm -0.0011): the fine-tune is mostly a cold-side re-calibration plus the pre_*/neighbour branch.
  * vs nn_final2 OOF (built with the leaky ALS): +0.0019 +- 0.0008 (cold +0.0016, warm +0.0020).
* Running at the time of writing: v8_nd_e3 (3 epochs, "longer run" lever) and v8_nd_gru (GRU + neighbour inputs).

### Longer train stage (synthesis lever iii)
v8_nd_e3 = v8_nd_q0 with 3 epochs (same seed, holdout-selected step 28000/29505): train-only **0.38733**, stage 2 **0.38725 / holdout 0.31288** (2 epochs: 0.38643 / 0.31175; +0.0008 val, +0.0011 holdout), n2c_pre fine-tune **0.38847** (best single fine-tune). Accepted; two more 3-epoch members (tf seed 42, GRU seed 43) are being added to nn_final3.
* v8_nd_gru (GRU + neighbour inputs, 2 epochs): train-only 0.38645 / 0.31177, stage 2 0.38718 / 0.31212, fine-tune 0.38802.
* v8_nd_e3_s42 (tf, 3 epochs): train-only 0.38700 / 0.31304, stage 2 0.38676 / 0.31281, fine-tune 0.38734.
* **v8_nd_gru_e3** (GRU, 3 epochs, seed 43): train-only **0.38816 / 0.31296**, stage 2 **0.38840 / 0.31306**, fine-tune **0.38898** (best single model of the project).
* nn_final3 with 19 members (adds v8_nd_e3, v8_nd_e3_s42, v8_nd_gru, v8_nd_gru_e3): fine-tuned ensemble **0.38870**, train-only **0.38763** (+0.2 x train-only: 0.38857, so `nn_final3` is now saved as the pure fine-tuned ensemble, like nn_final2).
* Queue started on GPUs N (coordinator's list): v8_nd_e5 (5-epoch probe), v8_nd_e3_s45 (tf), v8_nd_din_e3 (DIN), v8_nd_deep_e3 (3 blocks + 2 history layers); member selection by holdout ensembles (`holdout_eval.py`).

### Member selection for the final (holdout ensembles, 08:45)
Equal-weight holdout ensembles: all 19 members 0.31235; the 6 neighbour-input (cvnd) members 0.31307 (+0.0007 +- 0.0003); the 3 cvnd 3-epoch members (v8_nd_e3, v8_nd_e3_s42, v8_nd_gru_e3) **0.31354 (+0.0012 +- 0.0003)**; all minus v2/DIN 0.31254. On val the 3-member 3-epoch ensemble equals the 19-member one (fine-tuned -0.0001 +- 0.0006, train-only +0.0002 +- 0.0006). Rule adopted (holdout-based, as the coordinator suggested "3-epoch members replace the 2-epoch ones"): **nn_final3 = equal-weight z-mean of all cvnd members trained for >= 3 epochs**. The 19-member version is kept as `nn_final3_wide_*` (fine-tuned 0.38870 / train-only 0.38763).

### Last probes and the final member set (09:10)
| member (cvnd inputs, seed) | epochs | train-only val / holdout | stage 2 val / holdout | n2c_pre fine-tune |
|---|---|---|---|---|
| v8_nd_e3 (tf, 23) | 3 | 0.38733 / 0.31281 | 0.38725 / 0.31288 | 0.38847 |
| v8_nd_e3_s42 (tf, 42) | 3 | 0.38700 / 0.31304 | 0.38676 / 0.31281 | 0.38734 |
| v8_nd_e3_s45 (tf, 45) | 3 | 0.38739 / 0.31230 | 0.38835 / 0.31227 | 0.38844 |
| v8_nd_gru_e3 (GRU, 43) | 3 | 0.38816 / 0.31296 | 0.38840 / 0.31306 | 0.38898 |
| v8_nd_din_e3 (DIN, 44) | 3 | 0.38448 / 0.31088 | 0.38524 / 0.31050 | 0.38588 |
| v8_nd_deep_e3 (tf, 3 blocks + 2 history layers, 46) | 3 | 0.38718 / 0.31308 | 0.38755 / 0.31316 | 0.38875 |
| **v8_nd_e5** (tf, 23) | **5** | **0.38970 / 0.31483** | 0.38879 / 0.31411 | stage-1 FT **0.38942** (from stage 2: 0.38926) |
* **Longer training keeps paying**: 2 -> 3 -> 5 epochs (same seed) gives holdout 0.31168 -> 0.31281 -> 0.31483 and val 0.38625 -> 0.38733 -> 0.38970; the holdout was still rising at the end of the 5-epoch run. Stage 2 (random-order feeds) no longer helps the 5-epoch model (holdout -0.0007), so the stage-1 model is used for it.
* Capacity: the deeper backbone (3 blocks, 2 history layers) is on par with the 3-epoch tf members (holdout +0.0001..+0.0009).
* Holdout ensembles: 6 three-epoch members + e5 (stage 1) 0.31408; + e5 (stage 2) 0.31398; 6 three-epoch members 0.31354; **e5 alone 0.31482** (better than any ensemble: the 5-epoch model dominates; more long-trained members are the obvious next step).

## Round 2 final deliverables (`<repo>/preds/`, float32, aligned with `data/flat/{val,test}/cand_*.npy`)
| file | what | val |
|---|---|---|
| **`nn_final3_val_oof.npy` / `nn_final3_test.npy`** | equal-weight z-mean of the n2c_pre fine-tunes of the 7 >=3-epoch cvnd members (6 x stage-2 3-epoch + stage-1 5-epoch) | **0.38958** OOF |
| **`nn_final3_trainonly_val.npy` / `nn_final3_trainonly_test.npy`** | same 7 train-stage models, **no val labels** (checkpoints selected on the train holdout) | **0.38879** |
| `nn_final3_wide_*` | previous 19-member version (all round-2 members + 2 v2 members) | 0.38870 / 0.38763 |
| `nn_v8_nd_e5_trainonly_*`, `nn_v8_nd_e5_n2cp_*` | best single model (5 epochs) | 0.38970 train-only / 0.38942 fine-tuned |
| `nn_<variant>_trainonly_{val,test}`, `nn_<variant>_{val_oof,test}`, `nn_<run>_test_folds` | every member / fine-tune (`_n2cp` = n2c_pre branch) | see tables |
| `preds/holdout/nn_<v>_holdout.npy` | holdout predictions used for member selection | |
Segment view of the final (val): vs nn_final2 OOF +0.0032 +- 0.0010 (cold-positive feeds +0.0039, warm +0.0026); train-only vs nn_final2_trainonly +0.0063 +- 0.0012 (cold -0.0228, warm +0.0141); fine-tune over train-only +0.0008 (cold +0.0116, warm -0.0024). vs nn_final3_wide: +0.0009 +- 0.0006 OOF, +0.0012 +- 0.0006 train-only.
Reproduction: `nn/run_final3.sh` documents the 19-member recipe; the final members are `train.py --xt cvnd --xt_drop 0 --epochs 3|5` (+ `--arch gru|din`, `--blocks 3 --hist_layers 2`) with the common v2 flags, then `EXTRA=n2c_pre SUF=n2cp bash stage2_ft_v3.sh <v>` (or `ft_list.sh v8_nd_e5` for the stage-1 fine-tune), then `ensemble.py --w_trainonly 0`.

## Round 2 summary and next steps
* New signal: leak-safe co-visitation + LUO popularity in the train stage (+0.003 val / +0.009 holdout), within-feed rank/z computed in-model (+0.001), neighbour/near-duplicate + candidate-competition context in the train stage (+0.0005 val / +0.0017 holdout), longer training (+0.0035 val / +0.003 holdout from 2 to 5 epochs), pretrained-encoder features in the fine-tune branch (+0.0015).
* Negative or neutral: hashed item-ID residual (-0.0022; cold-positive feeds -0.0075), history-only item IDs (same 2nd-epoch collapse), test pseudo-labels (-0.0004), ID-input dropout (trades warm for cold, holdout prefers q = 0), category-intent head, clean ALS in the fine-tune branch (~0), item cross-feed context ctx_ix (deleted upstream).
* Leak/robustness fixes: n2clean preset (no non-LUO it_hist, no in-split counts, no leaky ALS); feature snapshot cache against upstream deletions; GPU guard extended to 4/6/7; CPU threads capped at 16 per process.
* Next (not started; coordinator asked not to put heavy jobs on GPU N while pretraining runs): (1) several 5+ epoch members (tf / GRU / deep, different seeds), stage-1 only, selected on the holdout; (2) the leak-free train-row `pre_*` as train-stage inputs with a missing indicator (report the holdout restricted to in-window feeds), once they land.

---------------------------------------------------------------------------------------------------------------------

# Round 3 (2026-11-11 09:13-12:05 UTC) - DONE: longer training, members at E*, train-stage pre_*
Launched in parallel (GPUs N and 6; all `--xt cvnd --xt_drop 0`, stage 1 only, holdout-selected, then n2c_pre fine-tune):
* **v9_e10**: epoch-horizon probe, tf, seed 23 (= v8_nd_e5), 10 epochs, evals every 2500 steps. The LR schedule is warmup + cosine over the *whole* horizon (to 3% of peak), so a 10-epoch run is not "5 epochs + 5 more": the LR is higher for longer.
* Members at 5 epochs (E* until the probe says otherwise): tf seeds 51 / 53, GRU seeds 52 / 54, deep (3 blocks + 2 history layers) seed 55, DIN seed 56.

Train-row `pre_*` (time-windowed encoders from the pretraining agent): at 09:13 only the first window (Sep 9-11, train days 20705-20707) was filled; the other windows were still being built (files rewritten later). Leak check on the filled window: pre_fe single-feature NDCG on the 6,952 holdout feeds of that window 0.2800 vs the NN (v8_nd_e5) 0.3153, ratio 0.888; on val feeds with M >= 50 (same feed-size regime) pre_fe 0.2792 vs NN 0.3114, ratio 0.897. Same level, so no leak signal. Preset `--xt cvnd_pre` = cvnd + 9 pre_* + `has:pre_fe` / `has:pre_idmax32` indicators, to be run when all windows exist.

### Round 3 interim (10:26)
| run (cvnd, stage 1) | epochs | best step | train-only val / holdout | n2c_pre fine-tune |
|---|---|---|---|---|
| v8_nd_e5 (round 2, seed 23) | 5 | 44000 | 0.38970 / 0.31483 | 0.38942 |
| v9_e5_s51 (tf) | 5 | 49175 | 0.38953 / 0.31417 | **0.39054** |
| v9_e5_s53 (tf) | 5 | 49175 | 0.38864 / 0.31408 | running |
| v9_din_e5_s56 (DIN) | 5 | 38000 | 0.38446 / 0.31201 | running |
| v9_e10 (tf, seed 23, probe) | 10 | (at 67500/98350) | 0.39049 / 0.31519 at 6.9 epochs, still rising | - |
GRU x2 and deep are still training (slower). Since the probe is still improving past 5 epochs, two 8-epoch tf members (seeds 57, 58) were added on GPU N.

### Epoch-horizon probe result (v9_e10, tf, seed 23, cosine over 10 epochs)
Holdout / val along the run (step: val / holdout): 10k 0.38323 / 0.30863; 20k 0.38541 / 0.31009; 30k 0.38758 / 0.31189; 40k 0.38765 / 0.31339; 50k 0.38952 / 0.31430; 60k 0.39088 / 0.31457; 70k 0.39062 / 0.31578; 80k 0.39091 / 0.31585; 90k 0.39152 / 0.31652; final 98350 **0.39141 / 0.31666** (best holdout = last step). vs the 5-epoch run with the same seed (v8_nd_e5 0.38970 / 0.31483): **+0.0017 val, +0.0018 holdout**; the holdout gain flattens in the last epoch (0.31650 -> 0.31666), so E* is about 10 epochs (maybe slightly more). Two more 10-epoch tf members (seeds 61, 62) were launched once GPUs N freed up; GRU at 10 epochs (~2x slower) does not fit the time budget.

### Train-stage pre_* leak check (all windows) and paired test (running)
All three windows are filled since 10:39 (Sep 9-15 and Sep 17-19; Sep 6-8 NaN; 22,177 of the 30,000 holdout feeds are in-window). pre_fe single-feature NDCG / NN (v8_nd_e5) NDCG on in-window holdout feeds: window 1 0.2800 / 0.3153 (ratio 0.888), window 2 0.2788 / 0.3168 (0.880), window 3 0.2755 / 0.3106 (0.887); val feeds with M >= 50: 0.2792 / 0.3114 (0.897). Same level as val, so no leak signal. Paired test: v10_pre3 = v8_nd_e3 config (3 epochs, seed 23) + 9 pre_* + has:pre_fe + has:pre_idmax32 (`--xt cvnd_pre`).

### Train-stage pre_* (`--xt cvnd_pre`): REJECTED - train/val inconsistency (likely leak in the train-row construction)
Paired with v8_nd_e3 (same config and seed 23), at equal steps (val / holdout):
| step | v8_nd_e3 | v10_pre3 (+ pre_* inputs) | delta val | delta holdout |
|---|---|---|---|---|
| 4000 | 0.37772 / 0.30440 | 0.37273 / 0.31039 | -0.0050 | +0.0060 |
| 10000 | 0.38289 / 0.30944 | 0.37619 / 0.31616 | -0.0067 | +0.0067 |
| 18000 | 0.38535 / 0.31085 | 0.38076 / 0.31932 | -0.0046 | **+0.0085** |
The holdout (train feeds, 74% of them in-window) improves a lot while val gets clearly worse. So the train-row pre_* carry label information that the val/test-row pre_* do not, and the model learns to rely on it. The single-feature check (ratio to the NN 0.88 vs 0.90 on val) did not show this; the model-based check does. Likely mechanism (to be verified by the pretraining agent): within a 3-day window a train user often has several feeds, and the encoder saw that user's earliest in-window snapshot, which for every feed except the earliest is a *later* snapshot of the same user (its history contains the items clicked in the earlier feeds). Val users rarely have more than one val feed, so val/test pre_* lack this. The run was stopped at 18k steps (no checkpoint); train-stage pre_* are not used. pre_* stay in the val/test fine-tune branch (n2c_pre), where they are clean.
* More round-3 members (train-only val / holdout; n2c_pre fine-tune): v9_e10 (10 ep, seed 23) 0.39141 / 0.31666, FT **0.39207**; v9_e10_s61 (10 ep) 0.39131 / 0.31674; v9_gru_e5_s52 0.38963 / 0.31501, FT 0.38902; v9_gru_e5_s54 0.38883 / 0.31471; v9_deep_e5_s55 0.38928 / 0.31479, FT 0.39069; v9_e5_s53 FT 0.38941; v9_din_e5_s56 FT 0.38528.
* Lead's note on train pre_* (scale mismatch: raw pre_fe sd 5.9 train vs 4.5 val; ID coverage 62% vs 46%): a second paired run, v10_prez3, uses pre_* only through within-feed z / rank (pre_fe_z, pre_new_z + in-model rank/z of pre_fe, pre_new, pre_hmax32; raw values dropped) + missing indicator.
* **v10_prez3** (pre_* only through within-feed z / rank + missing indicator, per the lead's scale-mismatch note), paired with v8_nd_e3, same pattern: step 4000 val 0.37307 vs 0.37772 (-0.0047), holdout 0.31086 vs 0.30440 (+0.0065); step 8000 val 0.37294 vs 0.38187 (-0.0089), holdout 0.31470 vs 0.30814 (+0.0066). Fails the acceptance rule (train-only val must improve); stopped at 8k steps. The problem is not the raw scale: within-feed ranks of the train-row pre_* are much more predictive on train feeds than the val-row pre_* are on val, i.e. the train-window and val/test constructions differ in information content (see the multi-feed-per-window hypothesis above). Train-stage pre_* are not used in nn_final4.

### Interim nn_final4 (11:25) and member selection so far
Holdout ensembles (stage-1 models): all 9 finished round-3 members (+ v8_nd_e5) 0.31662; without DIN 0.31671; **the two 10-epoch members (v9_e10, v9_e10_s61) 0.31761 (+0.0010 +- 0.0003 over all)**. The longest-trained models dominate, as in round 2.
Interim `nn_final4_*` = the two 10-epoch members: fine-tuned OOF **0.39301**, train-only **0.39246** (nn_final3: 0.38958 / 0.38879). Still running: v9_e10_s62, v9_e10_s63 (10 epochs), v9_e8_s57 / s58 (8 epochs); the final rule picks among the pre-declared sets {all, >= 8 epochs, 10 epochs, all minus DIN} by holdout ensemble NDCG.

## Round 3 final (12:05)
All round-3 members (cvnd inputs, stage 1 only, holdout-selected checkpoint, then n2c_pre fine-tune):
| member | epochs | train-only val / holdout | fine-tuned OOF |
|---|---|---|---|
| v9_e10 (tf, seed 23) | 10 | 0.39141 / 0.31666 | **0.39207** |
| v9_e10_s61 (tf) | 10 | 0.39131 / 0.31674 | 0.39168 |
| v9_e10_s62 (tf) | 10 (best step 85000) | 0.39072 / 0.31620 | 0.39108 |
| v9_e10_s63 (tf) | 10 (best step 97500) | **0.39165** / 0.31648 | 0.39006 |
| v9_e8_s57 (tf) | 8 | 0.39034 / 0.31617 | 0.39092 |
| v9_e8_s58 (tf) | 8 | 0.38938 / 0.31604 | 0.38993 |
| v9_e5_s51 (tf) | 5 | 0.38953 / 0.31417 | 0.39054 |
| v9_e5_s53 (tf) | 5 | 0.38864 / 0.31408 | 0.38941 |
| v9_gru_e5_s52 (GRU) | 5 | 0.38963 / 0.31501 | 0.38902 |
| v9_gru_e5_s54 (GRU) | 5 | 0.38883 / 0.31471 | 0.38931 |
| v9_deep_e5_s55 (3 blocks, 2 hist layers) | 5 | 0.38928 / 0.31479 | 0.39069 |
| v9_din_e5_s56 (DIN) | 5 (best step 38000) | 0.38446 / 0.31201 | 0.38528 |
(GRU/DIN at 10 epochs did not fit the time budget: GRU trains ~2x slower.)

**Member selection (pre-declared candidate sets, equal weights, chosen by holdout ensemble NDCG, no val):** all 13 (incl. v8_nd_e5) 0.31730; >= 8 epochs (6) 0.31799; **10 epochs (4) 0.31810**; all minus DIN 0.31749. Chosen: the four 10-epoch tf members.

**`nn_final4`** (`preds/nn_final4_{val_oof,test,trainonly_val,trainonly_test}.npy`; nn_final3 untouched):
| | val | cold-positive feeds | warm-positive feeds |
|---|---|---|---|
| `nn_final4_trainonly` (no val labels) | **0.39281** | 0.35464 | 0.39881 |
| `nn_final4` (fine-tuned OOF) | **0.39322** | 0.36862 | 0.39536 |
| vs nn_final3 OOF | +0.0036 +- 0.0008 | +0.0068 +- 0.0020 | +0.0030 +- 0.0010 |
| vs nn_final3_trainonly | +0.0040 +- 0.0008 | +0.0045 +- 0.0020 | +0.0041 +- 0.0010 |
| fine-tune over train-only | +0.0004 +- 0.0007 | +0.0140 +- 0.0018 | -0.0035 +- 0.0009 |
For reference, the >= 8-epoch set (6 members) gives 0.39294 OOF / 0.39250 train-only. With 10-epoch training the val fine-tune is almost neutral overall (a cold/warm re-calibration plus the pre_*/neighbour branch); downstream can use either the fine-tuned or the train-only ensemble.

Round-3 conclusions:
* **Training length is the lever**: cosine to 10 epochs gives +0.0017 val / +0.0018 holdout over 5 epochs (same seed), consistently over 4 seeds (train-only 0.3907-0.3917), and the gain covers cold and warm feeds. The holdout was still creeping up at 10 epochs (flattening), so 12-15 epochs might add a little more.
* Ensembles: the longest-trained members dominate (mixing in 5-epoch or other-architecture members lowers the holdout ensemble).
* **Train-stage pre_*: rejected twice** (raw inputs, and within-feed z/rank only): holdout +0.006..+0.009 but train-only val -0.005..-0.009 at equal steps, so the train-row pre_* carry information the val/test-row pre_* do not (see the multi-feed-per-window hypothesis). The lead's single-feature leak check passes, but the paired model check fails the acceptance rule.
* Compute: GPUs N only; torch threads capped at 16 per process; GRU/DIN/deep 5-epoch runs ~85-120 min under sharing, tf 10-epoch ~35-90 min depending on sharing.

---------------------------------------------------------------------------------------------------------------------

# Round 4 (started 2026-11-11 12:05 UTC) - IN PROGRESS
(Lead: the w_trainonly = 0.5 mix of nn_final4 = 0.39401 val, now submission sub_v10. Lead's correction on the train-stage pre_* explanation: a user's earliest in-window snapshot precedes their later feeds, so the multi-feed hypothesis cannot be the mechanism; diagnosis pending with the pretraining agent.)
Launched at 12:05 (all `--xt cvnd --xt_drop 0`, stage 1, holdout-selected, n2c_pre fine-tune via `ft_list.sh`):
* lane: GRU 10 epochs x2 (seeds 71, 72), capacity probe d=256 / 8 heads at 10 epochs (seed 23, paired with v9_e10).
* lane: horizon probes at 15 and 20 epochs (seed 23, evals every 5000 steps), tf 10 epochs seed 73, gated attention (`--attn_gate`: elementwise sigmoid output gate in all 5 attention modules, nnx A1 re-implemented in nn/model.py) at 10 epochs seed 23 (paired).
* lane: tf 10 epochs seeds 74, 75, 76; deep (3 blocks + 2 history layers) 10 epochs seed 77.

### Round 4 re-prioritisation (12:37): nnx winners ported into nn/
Ported (all flags default off; old configs reproduce exactly):
* `train.py --muon` (torch.optim.Muon with `adjust_lr_fn="match_rms_adamw"`, momentum 0.95 Nesterov, for every hidden 2-D Linear weight; AdamW for the rest; k/v projections split automatically (`split_kv`, initialised from the fused kv)), `--sched wsd --cooldown_frac 0.2` (warmup-stable-decay with 1-sqrt cooldown to 3%), `--cooldown_mix 0.3` (exploration-feed batches during the cooldown), `--aux_anneal 0.1` (aux weight annealed with the cooldown lr).
* `--retr_m 16` (TabR-style candidate-conditioned retrieval over the top-16 history items by raw cosine, zero-init output) and `--attn_gate` (elementwise sigmoid output gate in every attention module); treatment-only modules are built under a forked RNG (`extra_init`) so baseline parameters keep their exact init.
* `finetune.py --decay cos --ema 0.99 --ema_start 0.5` (the nnx "fta" recipe; saved OOF/test = final EMA weights).
Paired 10-epoch tests vs v9_e10 (seed 23, cvnd, stage 1, holdout-selected) launched: v12_muwsd_e10 (Muon lr 2e-3 + WSD + cooldown mix + aux anneal), v12_rg_e10 (retr16 + gate), v12_all_e10 (all combined). To make room, the old cosine/AdamW 15/20-epoch probes and two plain 10-epoch seeds (75, 76) were stopped; still running: GRU x2, d256, gate-only, tf seeds 73/74, deep.

### Round 4 interim (13:10)
Paired 10-epoch curves vs v9_e10 (seed 23; val / holdout at equal steps):
| step | v9_e10 (base) | (a) muon2e-3+wsd | (b) retr16+gate | (c) all combined |
|---|---|---|---|---|
| 2500 | 0.37720 / 0.30271 | 0.38156 / 0.30645 | 0.37960 / 0.30445 | 0.38328 / 0.30897 |
| 5000 | 0.38044 / 0.30514 | 0.38239 / 0.30742 | 0.38278 / 0.30815 | 0.38560 / 0.31098 |
| 10000 | 0.38323 / 0.30863 | 0.38511 / 0.30961 | 0.38592 / 0.31127 | **0.38943 / 0.31335** |
(c) leads by about +0.006 val / +0.005 holdout at 10k steps (WSD keeps the peak lr until 80% of training, so the final comparison is what counts). Two more (c) seeds (81, 82) and two (c) + prew members (83, 84) were started speculatively so that nn_final5 can finish within the time budget. Stopped to free GPUs (all plain round-3 recipe): tf seeds 73 / 74 at 45k steps (0.38854 / 0.31340, 0.38822 / 0.31251), deep at 35k, gate-only at 42.5k (0.38918 / 0.31477, about +0.0015 / +0.0013 over v9_e10 at the same step), the d256 capacity probe at 27.5k (0.38549 / 0.31132: no gain visible vs d=128) and the old cosine 15/20-epoch probes.

prew_* train-stage paired test (v12_prew3 = v8_nd_e3 config + prew_fe_z, prew_new_z, in-model rank/z of prew_fe / prew_new / prew_hmax32, has:prew_fe; 3 epochs, seed 23), val / holdout at equal steps:
| step | v8_nd_e3 | v12_prew3 |
|---|---|---|
| 4000 | 0.37772 / 0.30440 | 0.37939 / 0.30481 |
| 8000 | 0.38187 / 0.30814 | 0.38339 / 0.30853 |
| 12000 | 0.38369 / 0.30885 | 0.38627 / 0.30946 |
| 16000 | 0.38513 / 0.31003 | 0.38620 / 0.31104 |
| 18000 | 0.38535 / 0.31085 | 0.38700 / 0.31120 |
Unlike the old pre_*, the leak-free prew_* improve val consistently (+0.0010..+0.0026) with the holdout flat-to-up: so far it passes the acceptance rule (final numbers below).
* **prew_* final (paired, holdout-selected checkpoints):** v12_prew3 val 0.38839 vs v8_nd_e3 0.38733 (**+0.0011 +- 0.0007**), holdout 0.31365 vs 0.31281 (**+0.0008 +- 0.0004**). Passes the acceptance rule (train-only val improves, holdout up); prew_* join the round-4 recipe (`--xt cvnd_prew`, members v12_allprew_s83 / s84).
* Time management (13:25): the Muon + retrieval runs are ~2x slower per step (retrieval top-16 gather + gate + split projections + Muon's Newton-Schulz), so the component tests (a) muon+wsd and (b) retr16+gate were stopped at 30k / 25k steps and GRU seed 72 at 32.5k, to let the nn_final5 members finish within the budget. Partial paired curves (val / holdout): (a) +0.0004 / +0.0005 over v9_e10 at 30k (the gain shrinks during the WSD plateau); (b) +0.0035 / +0.0031 at 25k; (c) +0.0037 / +0.0048 at 20k. Remaining: (c) seed 23 (paired test and member), (c) seeds 81 / 82, (c) + prew seeds 83 / 84, GRU seed 71 (round-3 recipe).
* 13:32: GRU seed 71 (round-3 recipe) stopped at 35k steps (0.38757 / 0.31276) because GPU N was the bottleneck for the two (c) runs there; GRU at 10 epochs does not fit the budget next to the slower (c) recipe.
* Lead (sub_v10 public 0.38786, val gains transfer >= 1:1): keep the plan; nn_final5 members get the nnx fta fine-tune (`--decay cos --ema 0.99 --ema_start 0.5 --epochs 4`, suffix `_n2cpf`) in addition to the standard n2c_pre snapshot fine-tune (`_n2cp`).
* Pre-declared nn_final5 candidate member sets (chosen by holdout ensemble NDCG only, equal weights): S1 all recipe-(c) members (v12_all_e10, v12_all_s81, v12_all_s82, v12_allprew_s83, v12_allprew_s84, v12_allprew_s85); S2 recipe (c) without prew (first three); S3 recipe (c) + prew (last three); S4 S1 + the four nn_final4 members (v9_e10, v9_e10_s61/s62/s63). Fine-tune: the fta recipe for every member (the plan); the n2c_pre snapshot version is reported alongside.

### Recipe (c) cooldown collapse at 10 epochs (15:00)
v12_all_e10 (seed 23) full tail (step: val / holdout / lr): 72500 0.39035 / 0.31655 / 2e-3; 75000 0.38911 / 0.31626; 77500 0.39270 / 0.31708; **cooldown starts at 78680**; 80000 0.39132 / **0.31723** (holdout-selected) / 1.5e-3; 82500 0.39121 / 0.31685; 85000 0.38849 / 0.31345; 87500 0.37882 / 0.30356; 90000 0.37281 / 0.29851; 92500 0.36869 / 0.29337; 95000 0.36651 / 0.29176; 97500 0.36400 / 0.28961; last 98350 **0.36426 / 0.28929**. Seeds 81-85 show the same collapse once they enter the cooldown. Holdout selection keeps the step-80000 checkpoint (0.39132 / 0.31723 = v9_e10's final 0.39141 / 0.31666 within noise), so as run, recipe (c) gives no 10-epoch gain: the WSD plateau at lr 2e-3 matches cosine, and the cooldown with 30% exploration-feed batches (~9 passes over the 134K-feed pool) destroys the model instead of polishing it. Hedges (coordinator request b, paired with v12_all_e10, seed 23): (i) lr 1e-3 + mix 0.3 started at 14:59; (iv, extra) lr 1e-3 + mix 0 started at 14:59, to separate the mixing from the lr; (ii) lr 2e-3 + mix 0.1 and (iii) lr 1e-3 + mix 0.1 start as soon as more slots free.

## Round 4 result: nn_final5 (15:50)
Members (recipe (c) = cvnd inputs + Muon lr 2e-3 + WSD (cooldown 20%, exploration mix 0.3, aux anneal 0.1) + retr16 + gated attention, 10 epochs, stage 1 only; holdout-selected checkpoints, all from the end of the WSD plateau / early cooldown because the cooldown collapses):
| member | seed | best step | train-only val / holdout | fine-tune n2c_pre snapshot | **fine-tune fta** (cos + EMA 0.99) |
|---|---|---|---|---|---|
| v12_all_e10 | 23 | 80000 | 0.39132 / 0.31723 | 0.39249 | 0.39385 |
| v12_all_s81 | 81 | 80000 | 0.39322 / 0.31811 | 0.39398 | 0.39480 |
| v12_all_s82 | 82 | 60000 | 0.39125 / 0.31778 | 0.39218 | 0.39329 |
| v12_allprew_s83 (+prew) | 83 | 70000 | 0.39266 / 0.31771 | 0.39411 | 0.39444 |
| v12_allprew_s84 (+prew) | 84 | 80000 | 0.39322 / 0.31793 | 0.39427 | 0.39474 |
| v12_allprew_s85 (+prew) | 85 | 80000 | 0.39312 / 0.31863 | (running) | 0.39495 |
fta beats the snapshot fine-tune on every member (+0.0005..+0.0014).

Cooldown curves (val / holdout from the last plateau eval; cooldown starts at step 78680; the selected step is never the last one):
* v12_all_e10: 77.5k 0.39270/0.31708, **80k 0.39132/0.31723**, 82.5k 0.39121/0.31685, 85k 0.38849/0.31345, 87.5k 0.37882/0.30356, 90k 0.37281/0.29851, 92.5k 0.36869/0.29337, 95k 0.36651/0.29176, 97.5k 0.36400/0.28961, last 0.36426/0.28929
* v12_all_s81: 77.5k 0.39200/0.31661, **80k 0.39322/0.31811**, 82.5k 0.39198/0.31634, 85k 0.38994/0.31390, 87.5k 0.37738/0.30146, 90k 0.37406/0.29733, 92.5k 0.36916/0.29339, 95k 0.36855/0.29308, 97.5k 0.36345/0.28823, last 0.36502/0.28963
* v12_all_s82 (selected 60k: 0.39125/0.31778): 77.5k 0.39177/0.31573, 80k 0.39212/0.31755, 82.5k 0.39109/0.31474, 85k 0.38984/0.31414, 87.5k 0.38044/0.30465, 90k 0.37237/0.29782, 92.5k 0.37060/0.29495, 95k 0.36825/0.29051, 97.5k 0.36684/0.28939, last 0.36606/0.28898
* v12_allprew_s83 (selected 70k: 0.39266/0.31771): 77.5k 0.39138/0.31723, 80k 0.39178/0.31676, 82.5k 0.39242/0.31682, 85k 0.38857/0.31367, 87.5k 0.37986/0.30480, 90k 0.37289/0.29686, 92.5k 0.36987/0.29382, 95k 0.36771/0.29232, 97.5k 0.36661/0.29125, last 0.36602/0.29070
* v12_allprew_s84: 77.5k 0.39398/0.31737, **80k 0.39322/0.31793**, 82.5k 0.39294/0.31724, 85k 0.39217/0.31484, 87.5k 0.37905/0.30294, 90k 0.37254/0.29699, 92.5k 0.37197/0.29644, 95k 0.36925/0.29214, 97.5k 0.36760/0.29049, last 0.36725/0.29015
* v12_allprew_s85: 77.5k 0.39225/0.31726, **80k 0.39312/0.31863**, 82.5k 0.39006/0.31494, 85k 0.38648/0.31389, ... 90k 0.37089/0.29657 (same collapse to the end)

**Member selection (holdout ensembles, pre-declared sets):** S1 all six (c) members **0.32162**; S2 (c) without prew 0.32054; S3 (c) + prew 0.32106; S4 S1 + the four nn_final4 members 0.32162 (-0.000003). Chosen: S1. Note the large ensemble gain: single members 0.3172-0.3186 holdout, ensemble 0.3216 (the members stop at different points of the plateau/cooldown, which adds diversity).

**`nn_final5`** (`preds/nn_final5_{val_oof,test,trainonly_val,trainonly_test}.npy`, equal-weight z-mean of the six fta fine-tunes / six train-only models; nn_final4 untouched) + convenience file `nn_final5_mix_{val_oof,test}` = 0.5 z(fine-tuned) + 0.5 z(train-only):
| | val | cold-positive | warm-positive |
|---|---|---|---|
| `nn_final5_trainonly` | **0.39712** | 0.35897 | 0.40357 |
| `nn_final5` (fta fine-tuned OOF) | **0.39784** | 0.37497 | 0.39971 |
| `nn_final5_mix` (0.5 / 0.5) | **0.39792** | 0.36724 | 0.40223 |
| fine-tuned vs nn_final4 | +0.0046 +- 0.0009 | +0.0064 +- 0.0025 | +0.0044 +- 0.0011 |
| train-only vs nn_final4_trainonly | +0.0043 +- 0.0009 | +0.0043 +- 0.0022 | +0.0048 +- 0.0011 |
(The snapshot-fine-tune version of the same set: 0.39676 on 5 members.)

Hedges (coordinator request b; paired with v12_all_e10, seed 23, fta fine-tune) still running at 15:50: (i) v13_lr1_mix3, (ii) v13_lr2_mix1, (iii) v13_lr1_mix1, plus (iv, extra) v13_lr1_mix0 to separate the exploration mixing from the lr. Results will be appended below.

### Hedge (ii) and (iv) results, nn_final6 / N2 launches (16:30)
* **(ii) lr 2e-3, mix 0.1** (v13_lr2_mix1, seed 23): no collapse. Cooldown (val / holdout): 77.5k 0.39270 / 0.31672 (plateau); 80k 0.39232 / 0.31700; **82.5k 0.39403 / 0.31899** (selected); 85k 0.39310 / 0.31821; 87.5k 0.39455 / 0.31836; 90k 0.39362 / 0.31768; 92.5k 0.39359 / 0.31723; 95k 0.39309 / 0.31557; last 0.39282 / 0.31605. Paired vs v12_all_e10's selected checkpoint: train-only val **+0.0027 +- 0.0010 (z 2.67)**, fta fine-tune 0.39598 vs 0.39385 **+0.0021 +- 0.0010 (z 2.25)**, holdout +0.0017 +- 0.0006.
* **(iv) lr 1e-3, mix 0** (v13_lr1_mix0): plateau 77.5k 0.39224 / 0.31750; cooldown 80k 0.39225 / 0.31638; 82.5k 0.39388 / 0.31840; **85k 0.39433 / 0.31845**; 87.5k 0.39377 / 0.31841; 90k 0.39331 / 0.31742; 92.5k 0.39285 / 0.31730. So without any exploration mixing the cooldown first helps (+0.002 val / +0.001 holdout over the plateau) and then slowly degrades too: the collapse at mix 0.3 is the pool, but the mild late decline has another cause (candidates: the aux anneal to 0.1, or Muon at low lr).
* Coordinator tasks: **nn_final6** = recipe (c) with lr 1e-3 + mix 0.1 (the preferred lr; (iii) itself is still running as seed 23) on 5 new seeds (v14_s92..s96) + v13_lr1_mix1; plus one lr 2e-3 / mix 0.1 / prew seed (v14_mix1_s91). **N2** = cooldown-only continuation of the six nn_final5 checkpoints (`--init v12_* --sched wsd --cooldown_frac 1.0 --warmup 0 --lr 2e-3 --cooldown_mix 0.1 --epochs 1`; verified: Muon re-created on the split k/v, lr decays 2e-3 -> 6e-5 from step 0, pool mixing from step 0): v15_cd_*. First run at 4000/9835 steps: 0.39269 / 0.31811 (start 0.39132 / 0.31723). New train.py diagnostics log the cooldown training loss separately for pool and regular batches (`cd_loss_reg` / `cd_loss_pool`); in v15_cd_all_e10 the pool loss falls faster (3.11 vs 3.29 at 4k steps).
* Guardrail P6 done: nn/feat_cache/train/pre_*.npy links removed, `cvnd_pre` / `cvnd_prez` presets deleted from train.py. `cvnd_pre2` preset added for W2 (pre2_fe_z, pre2_new_z, in-model rank/z of pre2_fe / pre2_new / pre2_hmax32 / pre2_id_contrib (parity fixed upstream), has:pre2_fe). GPU guard now allows 4-7.

### N2: cooldown-only continuation of the nn_final5 plateau checkpoints (17:05)
`train.py --init v12_* --muon --lr 2e-3 --sched wsd --cooldown_frac 1.0 --warmup 0 --cooldown_mix 0.1 --aux_anneal 0.1 --epochs 1` (lr 1-sqrt from 2e-3 to 6e-5 over one epoch, 10% exploration batches), holdout-selected, fta fine-tune:
| member | start (plateau) train-only val / holdout / fta | cooled best step | cooled train-only val / holdout | cooled fta |
|---|---|---|---|---|
| all_e10 | 0.39132 / 0.31723 / 0.39385 | 7000 | 0.39318 / 0.31927 | 0.39520 |
| all_s81 | 0.39322 / 0.31811 / 0.39480 | 7000 | 0.39343 / 0.31859 | 0.39541 |
| all_s82 | 0.39125 / 0.31778 / 0.39329 | 8000 | 0.39466 / 0.31951 | 0.39620 |
| allprew_s83 | 0.39266 / 0.31771 / 0.39444 | 9835 | 0.39426 / 0.32014 | 0.39584 |
| allprew_s84 | 0.39322 / 0.31793 / 0.39474 | 7000 | 0.39451 / 0.31903 | 0.39658 |
| allprew_s85 | 0.39312 / 0.31863 / 0.39495 | 7000 | 0.39555 / 0.31989 | 0.39613 |
Every member improves (train-only +0.0002..+0.0034, fta +0.0006..+0.0029, holdout +0.0005..+0.0024). The cooldown peaks around 70% of the continuation and then declines; the pool training loss falls far below the regular loss late in the run (v15_cd_all_e10 at 8k steps: pool 2.43 vs regular 2.79), i.e. the 134K-feed exploration pool is being memorised even at mix 0.1.
**Ensemble level** (pre-declared holdout sets): plateau six (nn_final5) 0.32162; **cooled six 0.32279 (+0.0012 +- 0.0004, chosen)**; cooled six + plateau six 0.32272 (+0.0011 +- 0.0003). On val, vs nn_final5 (mix 0.39792 / fta 0.39784 / train-only 0.39712):
* cooled six (`nn_final5cd_*`): mix 0.39828 (+0.0004 +- 0.0006), fta 0.39846 (+0.0006 +- 0.0006), train-only 0.39702 (-0.0001 +- 0.0007); M<50 +0.0008, M>=50 +0.0000 (mix).
* cooled six + plateau six (`nn_final5cd12_*`): mix **0.39870 (+0.0008 +- 0.0004)**, fta 0.39837, train-only 0.39818; M<50 +0.0012, M>=50 +0.0005.
Neither passes the "> 2 SE on val vs nn_final5 mix" bar: better single models do not translate into a better ensemble, as the lead predicted (the plateau members gain more from ensembling). Files saved for the lead's decision; nn_final5 stays the reference.
Standard slices (new `nn/pcmp.py`; val Tue/Thu by local day; "late" = latest 20% of holdout feeds by timestamp):
* per model (6 pairs pooled), train-only val +0.0018 +- 0.0005 (M<50 +0.0018, M>=50 +0.0018, Tue +0.0020, Thu +0.0016); holdout +0.0015 +- 0.0003 (late +0.0017 +- 0.0007).
* cooled-six mix vs nn_final5 mix: +0.0004 +- 0.0006 (M<50 +0.0008, M>=50 +0.0000, Tue +0.0004, Thu +0.0003); holdout ensemble +0.0012 +- 0.0004 (late +0.0011 +- 0.0009).
* cooled6+plateau6 mix: +0.0008 +- 0.0004 (M<50 +0.0012, M>=50 +0.0005, Tue +0.0009, Thu +0.0007); holdout +0.0011 +- 0.0003 (late +0.0007 +- 0.0006).

### W2: paired train-stage test of pre2 (cvnd_pre2 vs cvnd, 3 epochs, holdout-selected, 2 seeds) (17:20)
`--xt cvnd_pre2` = cvnd + pre2_fe(_z), pre2_new(_z), pre2_hmax32, pre2_id_contrib (fixed parity), has:pre2_fe; in-model rank/z on the raw four. All other args identical to v8_nd_e3 / v8_nd_e3_s42.
| seed | cvnd train-only val / holdout | cvnd_pre2 train-only val / holdout | delta val | delta holdout |
|---|---|---|---|---|
| 23 | 0.38733 / 0.31281 | 0.38877 / 0.31314 | +0.00143 | +0.00033 |
| 42 | 0.38700 / 0.31304 | 0.38805 / 0.31310 | +0.00105 | +0.00006 |
Pooled per feed: val **+0.00124 +- 0.00061 (z 2.0)**; M<50 +0.0030 +- 0.0012, M>=50 +0.0000 +- 0.0006, Tue +0.0017, Thu +0.0007. Holdout +0.0002 +- 0.0003 (late 20% -0.0003 +- 0.0008). Two-seed ensembles: val +0.0015 +- 0.0007, holdout -0.0001 +- 0.0004.
Under the unified rule this is a marginal pass (val just over 2 SE, holdout >= 0), but the whole gain sits in M<50 feeds, which the M>=50-only holdout cannot check; on M>=50 it is exactly 0 on val and ~0 on the holdout.

### v13 hedges at 10 epochs (seed 23, cvnd, Muon + WSD cooldown_frac 0.2, retr+gate), pre-declared nn_final6 recipe choice (17:35)
Each arm is holdout-selected. In the table, train-only val / holdout / fta is the score at the selected step.
| arm | selected step | train-only val / holdout / fta | end of run (98k) val / holdout |
|---|---|---|---|
| (i) lr 1e-3, mix 0.3 | 45000 (plateau) | 0.39149 / 0.31822 / - | 0.36514 / 0.29147 (collapse from 82.5k) |
| (ii) lr 2e-3, mix 0.1 | 82500 | 0.39403 / 0.31897 / **0.39598** | 0.39282 / 0.31605 |
| (iii) lr 1e-3, mix 0.1 | 82500 | 0.39389 / 0.31828 / 0.39466 | 0.38965 / 0.31338 (holdout -0.005 after the peak) |
| (iv) lr 1e-3, mix 0 | 85000 | **0.39433** / 0.31844 / 0.39475 | 0.39361 / 0.31751 (mild decline) |
Paired comparisons (all slices from pcmp.py):
* (iii) vs (ii): train-only -0.0001 +- 0.0010 (M<50 +0.0026, M>=50 -0.0020); holdout -0.0007 +- 0.0006 (late -0.0014); fta -0.0013 +- 0.0010 (M>=50 -0.0035).
* (iv) vs (iii): train-only +0.0005 +- 0.0010; holdout +0.0002 +- 0.0006 (late +0.0014 +- 0.0013); fta +0.0001 +- 0.0009.
* (iv) vs (ii): train-only +0.0003 +- 0.0011; holdout -0.0005 +- 0.0006; fta -0.0012 +- 0.0010.
* vs v12_all_e10 (same seed): train-only (ii) +0.0027, (iii) +0.0026 +- 0.0011, (iv) +0.0030 +- 0.0011. fta (ii) +0.0021 +- 0.0010, (iii) +0.0008 +- 0.0010, (iv) +0.0009 +- 0.0010.
**Rule application.**
1. lr: 2e-3 does not win by > 2 SE on train-only val (+0.0001), so lr 1e-3.
2. Mix: (iv) mix 0 is below the best mix-0.1 arm on the holdout ((ii), -0.0005), so mix 0.1. The recipe is (iii) = the running v14_s92..s96 + v13_lr1_mix1.
If "best mix-0.1 arm" is read as the best lr-1e-3 mix-0.1 arm, (iv) qualifies (holdout +0.0002, val +0.0005) and mix 0 would be preferred.
Observation: both lr-1e-3 arms gain little from the fine-tune (+0.0004/+0.0008 over train-only), whereas lr 2e-3 gains +0.0020. On fta, (ii) is the best arm (z 1.4 vs (iii); single FT seed).

### W3: re-FT hygiene preset n2c_pre_v2 + user-0 singleton folds (17:55)
New FT preset `n2c_pre_v2` (finetune.py): n2c_pre with pre_id_contrib -> pre_id_contrib_v2, and without hour, it_tr_*, u_tr_* and it_cand_cnt_train (41 extra features on disk vs 51). New flag `--u0_singleton`: the 658 user-0 val feeds become singleton GroupKFold groups. Existing presets are unchanged. Applied with fta to the six nn_final5 members (`*_n2cpv2`) and compared with their n2c_pre fta (`*_n2cpf`):
* per member: -0.0002, -0.0001, -0.0005, -0.0013, -0.0006, +0.0000; pooled -0.00045 +- 0.00023 (M<50 -0.0004, M>=50 -0.0005, Tue -0.0003, Thu -0.0006).
* fta ensemble (`nn_final5v2_val_oof`) 0.39764 vs 0.39784: **-0.00020 +- 0.00041** (M<50 +0.0003, M>=50 -0.0006 +- 0.0004, Tue +0.0004, Thu -0.0009 +- 0.0006).
* family mix (`nn_final5v2_mix_val_oof`) 0.39805 vs 0.39792: +0.00013 +- 0.00028 (all slices within 1 SE).
Pre-declared non-inferiority rule (ensemble delta >= -0.0003, no slice below -2 SE): **W3 passes**, on both the fta ensemble and the family mix. The member-level loss is -0.00045 (z -2.0), and the ensemble absorbs most of it.

### FR-1 fix and W2 confirmation: cvnd_pre2s (18:15)
The feature review found two problems with cvnd_pre2. Raw pre2_fe and pre2_new (median ~15) pass through the +-10 clip in attach_xt, so 84-92% of the present rows tie at 10 and the in-model rank channel is almost dead. The preset also contained pre2_id_contrib (density-skewed). New preset **cvnd_pre2s** = cvnd + pre2_fe_z, pre2_new_z, pre2_fe_s, pre2_new_s, pre2_hmax32, has:pre2_fe. The _s columns are clip-safe (x+20)/6 copies, hard-linked into nn/feat_cache under new names. XT_REL = seq_cv_invd20, it_hist_all_luo, pre2_fe_s, pre2_new_s, pre2_hmax32. No id_contrib.
Paired with v8_nd_e3 / v8_nd_e3_s42 (3 epochs, holdout-selected, only xt differs):
| seed | cvnd train-only / holdout | cvnd_pre2s train-only / holdout | delta val | delta holdout |
|---|---|---|---|---|
| 23 | 0.38733 / 0.31281 | 0.38891 / 0.31340 | +0.00158 | +0.00059 |
| 42 | 0.38700 / 0.31304 | 0.38924 / 0.31287 | +0.00224 | -0.00017 |
Pooled: val **+0.00191 +- 0.00058 (z 3.3)**; M<50 +0.0040 +- 0.0011, M>=50 +0.0005 +- 0.0006, Tue +0.0027, Thu +0.0011. Holdout +0.0002 +- 0.0003 (late +0.0003 +- 0.0008). Two-seed ensembles: val +0.0018 +- 0.0007, holdout +0.0001.
vs the broken cvnd_pre2: val +0.0007 +- 0.0005, holdout +0.0000. **Passes the unified rule.** The gain is still concentrated in M<50, but M>=50 is now positive too. nn_final7 has been launched with cvnd_pre2s: v17_pre2s_s101..106, recipe (ii), on GPUs N only, with plateau snapshots at 60k/70k/77.5k.

v14_s96 (nn_final6 member, recipe (iii)) selected its plateau checkpoint at 47.5k (train-only 0.39180 / holdout 0.31788): the lr-1e-3 mix-0.1 cooldown peaked at 0.31773 and then fell to 0.31342.

### W3b: preset n2c_pre_v3 (on top of n2c_pre_v2) + user-0 singleton folds (18:22)
`n2c_pre_v3` makes these changes (42 features):
* prunes the 14 phantom names;
* drops seq_cv_norm;
* adds pre_idmax32_has;
* pre_hmax32_c / pre_hmax8_c (-2 sentinel -> NaN) plus one pre_hmax32_sent flag;
* it_hist_all_luo_z -> it_hist_all_luo_zsl (signed log1p).
The derived files come from `nn/build_w3b_feats.py`. Run on the six nn_final5 members with fta and --u0_singleton (`*_n2cpv3`):
* per member vs n2c_pre: -0.0001, -0.0003, -0.0002, -0.0011, -0.0003, +0.0002; pooled -0.00030 +- 0.00023 (M<50 -0.0004, M>=50 -0.0003, Tue +0.0000, Thu -0.0007).
* fta ensemble (`nn_final5v3_val_oof`) 0.39762 vs 0.39784: **-0.00022 +- 0.00042** (M<50 +0.0002, M>=50 -0.0005 +- 0.0005, Tue +0.0004, Thu -0.0009 +- 0.0006, z -1.5).
* family mix (`nn_final5v3_mix_val_oof`) 0.39816 vs 0.39792: +0.00024 +- 0.00029 (M<50 +0.0007, M>=50 -0.0000, Tue +0.0001, Thu +0.0004).
* vs W3: ensemble -0.00001 +- 0.00017; per member +0.00014 +- 0.00010.
**W3b passes the pre-declared non-inferiority rule. Since W3 passes too, the rule prefers W3b.** The n2c_pre_v3 + u0_singleton fta (`_n2cpv3`) is therefore added for every nn_final6 / nn_final7 member, next to the n2c_pre fta.
