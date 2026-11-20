# Review: NN implementation and math (nn/model.py, data.py, train.py, finetune.py, predict.py, ensemble.py)

Reviewer: critical-review workflow, dimension "nnmath". Date: 2026-11-11 ~14:30 UTC. CPU only, read-only on project files.
Scripts and outputs: `/tmp/scratchpad/review/nnmath/`
(`harness.py` is a CPU copy of `SplitData.gather` / `attach_xt` that writes nothing to the project; `t*.py` are the tests).

Live round-4 recipe as actually launched (from `ps`, PIDs 3564745, 3580382, 3580383, 3611554, 3611555, 3632343):
`--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout --loss_norm feed --xt cvnd|cvnd_prew --epochs 10 --muon --lr 2e-3 --sched wsd --cooldown_frac 0.2 --cooldown_mix 0.3 --aux_anneal 0.1 --retr_m 16 --attn_gate`, each chained to `EXTRA=n2c_pre SUF=n2cp bash ft_list.sh <variant>`.

**Bottom line.** I found no leak and no masking bug in the NN. Masking, set equivariance, independence from other feeds in the batch, the loss, the aux direction, the WSD, Muon, EMA and fold mechanics are all correct (see "Checked and correct" below). There are seven verified issues, and only one of them is likely to cost real NDCG: the queued fine-tunes will not use the fta recipe (NNMATH-1). The others are math inconsistencies with measured or expected impact at noise level. The two "suspected" items are diagnostics, not bugs.

## Findings (ranked by severity)

### NNMATH-1 (medium, verified): the queued v12 fine-tunes will run WITHOUT the fta recipe
- The six live train jobs end with `EXTRA=n2c_pre SUF=n2cp bash ft_list.sh <v>`. `nn/ft_list.sh` (edited 13:54:45) runs `finetune.py ... --lr 5e-5 --epochs 6 ${FTARGS:-}`. The chained commands do not set `FTARGS`, and `/proc/<pid>/environ` of the six `sh -c` parents (3564741, 3580378, 3580380, 3611550, 3611552, 3632341) has no `FTARGS`.
- So every v12 Muon/WSD member will get the old recipe: constant lr, 6 epochs, snapshot mean, no EMA. It will not get `--decay cos --ema 0.99 --ema_start 0.5` (and nnx used `--epochs 4`).
- Expected cost: nnx_exp_optim.md measured fta as a WIN specifically on Muon-pretrained models: +0.00112 (C3a row), and +0.0021 ± 0.0007 vs the base fine-tune in the combo table. On AdamW models it was neutral.
- Fix: before the chains reach the fine-tune (ETA 15:45-17:00), either export the arguments in the files the chains read, or re-run the fine-tune for each v12 checkpoint by hand with `FTARGS="--decay cos --ema 0.99 --ema_start 0.5 --epochs 4"`, writing to a different SUF. Do NOT edit ft_list.sh while a chain may be inside it: bash reads scripts incrementally. If both fine-tunes exist, compare them paired on OOF; that is one coarse choice.

### NNMATH-2 (low-medium, verified math bug): the aux down-weighting by (1 - p_random) cancels in pure exploration batches
- `train.py:300-301`: `fw_aux = 1 - prand[fidx]`, then `loss += aux_w * (CE_aux * fw_aux).sum() / fw_aux.sum()`.
- Dividing by the batch's own weight sum makes this a weighted mean. In a batch where every feed is random-like, the weights cancel and the aux CE (toward `softmax(-pos/10)` of a *random* display order) gets its full weight `aux_w(s)`.
- `--cooldown_mix 0.3` draws exactly such batches: 30% of the cooldown steps (about 5,900 steps; the pool of 134,118 feeds gives 644 batches per pass).
  - In those batches the intended factor is mean(1 - p) = 0.159. The applied factor is 1.0, about 6x (t8_aux.py).
  - Over the cooldown, aux_w anneals 0.5 → 0.1, so the trunk receives aux gradients toward noise targets.
- Impact: not measurable on CPU. Likely small, because aux is annealed and the aux head absorbs most of the effect. It is still a wrong objective during exactly the phase that produces the selected checkpoint.
- Fix (one line): `/ fw_aux.numel()`, or `/ a.max_feeds` to match `loss_norm=feed`. Test it as a paired 10-epoch run (holdout and train-only val).

### NNMATH-3 (low, verified): aux and main loss are normalised differently, so the effective aux weight grows with feed length
- Main loss (`loss_norm=feed`) = sum over feeds / 512. Aux = per-batch weighted mean.
- Batches hold B = min(512, 32768 // Mpad) feeds, so the per-feed aux:main ratio is multiplied by 512/B:
  - 1x for M ≤ 64;
  - 3.5x at the 90th percentile and 4.5x at the 99th (M ≈ 280);
  - 1.61x on average over training feeds (41% of feeds sit in batches with B < 512) (t8_aux.py).
- In effect "aux_pos 0.5" means 0.5 on short feeds and about 2.2 on long ones. The fix is the same as NNMATH-2: normalise aux by the same constant denominator. Impact unknown, probably small. Test it together with NNMATH-2.

### NNMATH-4 (low, verified): dense features are fp32 in training but bf16 at inference
- Training and fine-tuning compute `X = model.dense(b)` outside autocast (`train.py:276`, `finetune.py:221`).
- `common.predict` / `eval_subset` call `model(b)` inside `torch.autocast(bfloat16)` (`common.py:51-52, 69-70`), so `dense_features` runs under autocast (`model.py:417-418`).
- The einsums then return bf16 (verified: `torch.einsum` under autocast gives `torch.bfloat16`). At val/test/holdout only, `cos_u`, `dot_u` and the candidate × history cosine matrix S are bf16-quantised, and with them `smax`, `smax8`, `smax32`, `smean`, `lse`, `top3`, the 10 per-event-type maxima, and the in-model z/pct of cos_u, smax, lse and top3. Quantisation also adds ties to the pct ranks.
- Measured on CPU over the full val (fp32 network, only the feature precision varied, t2_featprec.py). NDCG(fp32 features) − NDCG(bf16 features):

| model | delta | SE |
|---|---|---|
| v9_e10 | +0.00016 | 0.00018 |
| v9_e10_s61 | −0.00020 | 0.00018 |
| v9_e10_s62 | +0.00021 | 0.00019 |
| v9_e10_s63 | +0.00013 | 0.00018 |
| v12_prew3 | +0.00009 | 0.00017 |
| 4-member equal-z ensemble | +0.00001 | 0.00019 |

  About 25% of feeds change their NDCG. The impact is at noise level.
- Fix (for consistency, no retraining needed, only re-prediction): in `predict` / `eval_subset`, compute `X = model.dense(b)` before entering autocast and call `model(b, X, use_pos=False)`.

### NNMATH-5 (low, verified): retrieval top-16 is chosen on bf16 cosines with arbitrary tie-breaking
- `Ranker.forward` computes `S = einsum(ce, he)` under autocast (bf16) and passes it to `HistRetrieval`, which takes `topk(16)` on it (`model.py:179`).
- On val candidates with H > 16 (t3c_retrties.py):
  - 29.7% have an exact tie at the 16/17 boundary even in fp32, because the same item repeats in the history. In bf16 the rate is 50.0%.
  - The selected set differs between fp32 and bf16 for 32.8%.
  - 26% of the 16 slots repeat an item that is already selected (t12_dups.py).
  - 41% of val history events repeat an item already in the same history.
- Tie-breaking depends on the tensor width: with the trained gate+retr checkpoint (`nn_x/attn/ckpt/attn_gate_retr_s23.pt`, which loads into nn/model.py), a feed scored alone vs inside a batch differs by up to 0.024 in logit when its H is in (16, 256) and Hx changes (t3d.py). Val/test batches almost always pad to Hx = 256, so in practice this does not fire.
- Measured NDCG impact on that model, full val (t4_retr_noise.py):
  - S fp32 vs bf16: +0.00008 ± 0.00015;
  - deterministic recency tie-break: +0.00001 ± 0.00016.
  Negligible as is. The duplicate crowding is more of an opportunity (see Ideas).

### NNMATH-6 (low, verified): GroupKFold puts all user-0 val feeds into one fold
- `finetune.py:183` groups by `user_id`, and user 0 is a pooled bucket. All 658 user-0 val feeds (4.6%) land in fold 0, making up 22.8% of that fold. Folds 1-4 have none (t6_folds.py).
- So the OOF score of each user-0 feed comes from a model that never saw a user-0 feed in fine-tuning. Each test user-0 feed is scored by the mean of 5 fold models, 4 of which were fine-tuned on all 658.
- Measured: fine-tuned minus train-only on user-0 val feeds is +0.0028 ± 0.0033 (non-user-0: +0.0003 ± 0.0007). OOF/test asymmetry is not significant at this size.
- Fix: `groups = np.where(user_id == 0, -(np.arange(n) + 1), user_id)`, i.e. user-0 feeds as singleton groups. This is consistent with the project rule that user 0 is not a real user. Val has 1.002 feeds per non-zero user, so GroupKFold matters only for user 0.

### NNMATH-7 (low / info, verified): optimizer grouping deviates from the stated rules
- AdamW weight decay: the filter `"emb" not in n` (`train.py:220`) excludes only `pos_emb`, `rev_emb` and `id_emb`. These all get wd = 0.02 (listed by groups.py):
  - the embedding tables `cat`, `et`, `es`, `rec`, `dom`, `retr.e_type`, `retr.e_src`, `retr.e_rec`;
  - `retr.beta` (a per-event-type logit prior);
  - `head.3` and `wide`.
- Muon set (46 matrices): it contains the input projections `c_lin` (256→d), `c_mlp.0` (n_dense→2d), `u_lin`, `h_lin` and `retr.wk`. The usual Muon recipe (Keller Jordan; Moonlight) keeps the first layer on AdamW.
- Neither choice is a bug. The combination was validated empirically (+0.004 to +0.006). Both are cheap one-flag ablations if anyone revisits the optimizer.
- Correct parts: 1-output layers (`head.3`, `aux_head.3`, `wide`), all norms, biases and embeddings are on AdamW. Torch Muon uses `adjust_lr_fn="match_rms_adamw"` (0.2·sqrt(max(A,B))) and decoupled wd with the base lr.

### NNMATH-8 (info, suspected): domain token at inference vs feed length
- `train.py:157` forces p_random = 0 for M < 50, so short train feeds always carry the production token. Val/test inference uses the pure random-like token for every feed (`infer_domain = 1`). That (short feed, random-like token) combination never occurs in training, yet 41% of val feeds have M < 50.
- CPU check with the four nn_final4 train-only members, production token minus random-like token (t10_domain.py):
  - M < 50: +0.00069, +0.00126, +0.00072, +0.00083 (SE ≈ 0.0009 each);
  - M ≥ 50: −0.00011 to +0.00023;
  - overall: +0.00027 ± 0.00027.
- Below 2 SE. The holdout has only M ≥ 50 feeds, so it cannot be used to select this. Do not change inference on this evidence. It points to a possible training-side fix: give short train feeds a prior p instead of 0.

### NNMATH-9 (info, suspected, other dimension): it_hist_*_luo raw level drifts with impression date
- Mean log1p(it_hist_all_luo) per candidate (t13_luo_shift.py):
  - train by day, Sep 6 → Sep 19: 2.92, 2.90, 2.86, ... 2.19, 2.22 (monotone decline);
  - val: 1.55;
  - test public: 1.42; private: 1.36.
- Against the checkpoint's train-fitted normalisation, the mean z is +0.05 on production train rows, −0.33 on holdout and −0.64 on val (t5_train.py).
- The LUO counts include other users' snapshots taken *after* the impression (`src/features_luo.py` counts all snapshots of all splits). Train rows therefore get more future cross-user popularity than val, and private test gets the least. The co-visitation features, by contrast, are causal (`feats_v2/seq_/feat_covis.py`).
- This is not an NN bug, and the normalisation itself is identical across splits. It is a raw-input shift that val may slightly overstate for private test. For the shift/leak reviewers.
- Possible fix: feed `it_hist_*_luo` only through the in-model within-feed rank/z (add them to `XT_RELONLY`), and accept or reject via paired train-only val.

(Not a finding: train.py was modified at 12:38:02, 40 s after `v12_all_e10` started at 12:37:22. That process runs the earlier file; s81/s82/s83-85 run the current one. I cannot tell whether the edit changed semantics.)

## Checked and correct (safe)
- **No attention across feeds:** each feed is its own row of a padded [B, M] batch. Set attention masks keys with `cmask`; target attention uses `[user, hmask]`.
  - CPU test (v9_e10, 15 val feeds including H = 0, H < 16, H = 256, M > 200): a feed scored alone vs in a batch differs by ≤ 7.6e-6 (t3b_batchdbg.py).
- **Padding invariance:** enlarging Mx/Hx and filling padded candidate and history slots with random values (embeddings, f1-f5, xt, categories, event type/source, cmiss) changes scores by ≤ 1.9e-6. Tested on a model with the live layout: retr16 + gate + split k/v, treatment weights perturbed so these paths are active (t3_invariance.py).
- **Permutation equivariance inside a feed:** ≤ 1.9e-6.
- **History masks:**
  - `hemb_ok` never marks padding (0 violations).
  - Retrieval fills invalid slots with −2 and gives them valid = False and a −inf logit. The zero-value sink keeps the softmax finite when H = 0 or H < 16, and 0 invalid slots were selected as valid.
  - Candidates without an embedding get only the sink, so their retrieval output is `Z.bias`.
  - The sim-bias mask uses −inf on padded keys. Neither the target-attention nor the hist-encoder key sets can be fully masked, because the user token is always valid.
- **In-model z / pct-rank:** computed per feed over `cmask` only, sd + 1e-4, ties get the mid-rank, (cnt − 1) is clamped. No −1 sentinel leaks into the stats.
- **Listwise loss:** padding is filled with −1e4 before log_softmax, the target is y/sum(y), feeds without positives get weight 0 (none exist in sampled train feeds: 0.000), and `feed` normalisation uses a constant denominator. Final logits are fp32 (`model.py:494-495`).
- **Aux target direction:** pos 0 = top.
  - Production-train CTR is 0.082 at pos < 5 vs 0.011 at pos ≥ 100.
  - `softmax(-pos/10)` puts the most mass on pos 0, and pos gaps (mean max pos + 1 − M = 8) are handled naturally.
  - Aux is used only in train.py; fine-tune and inference ignore `aux_out`.
  - The anneal is correct: 0.5 → 0.30 at 25% of the cooldown → 0.10 at the last step (t11_init.py).
- **WSD:** warmup of 500 steps, plateau to step 78,680 (= 0.8 × 98,350), 1 − sqrt cooldown reaching exactly 3.00% of the base lr at the last step.
- **Paired init:** the muon / retr / gate model shares every baseline parameter bit-for-bit with the plain model (0 differing tensors). The split k/v weights equal the halves of the fused kv, and `retr.Z` starts at zero (t11_init.py).
- **Cooldown pool:** it excludes the holdout (134,118 = 164,118 − 30,000) and is reshuffled when exhausted. Exploration batches get the same p-mixed domain token as in normal training.
- **Domain flag:**
  - train.py, predict.py and test prediction use the random-like token.
  - The fine-tune copies `dom[1]` into `dom[2]` and uses 2, so the fine-tune's epoch 0 equals the train-only model.
  - Holdout evaluation uses the same p-mix as training.
- **user0 flag:** `user_id == 0` in every split (train 3.9%, val 4.6%, test 4.1%).
- **Holdout:** the same rule and seed (12345) in train.py and holdout_eval.py. It is excluded from the training feeds, from `feature_stats` and from the cooldown pool. CPU NDCG on 3,000 of the 30,000 holdout feeds is 0.3158, vs 0.3167 logged for the full holdout.
- **Normalisation:**
  - `f_mean` / `f_std` come from 40 training batches in eval mode, are stored in the checkpoint buffers and are reused unchanged for val, test and fine-tune. Min f_std is 0.0054, so no 1e-3 clamp is active.
  - The xt transforms are fixed and split-independent.
  - NaN becomes 0. NaN rates agree across splits: ctx_* 0.3-4.4% (no history or user embedding); prew_* 38-45%, with a `has:` indicator. No infs.
  - The ±6 clamp mostly hits the −1.5 no-history sentinel of the similarity features: 1.4% of val rows, 2.7% of holdout rows, 1.2% of production rows.
- **Reproduction:** the CPU re-implementation of gather, attach_xt and predict reproduces the saved GPU val NDCG within 0.0003 for v9_e10, s61, s62, s63 and v12_prew3.
- **Scores and ties:** saved scores are float32. Exact within-feed ties: 0-115 adjacent pairs out of 1.39M rows. The NDCG gap between pessimistic and optimistic tie handling is ≤ 1e-6 for nn_final4, its members and v12_prew3.
- **Inference mode:** predict and eval call `model.eval()` (no dropout) and restore `train()` afterwards. Batch order is deterministic, and scatter writes valid positions only.
- **Fine-tune mechanics:**
  - EMA `lerp(θ̄, θ, 1 − β)` starts at 0.5 epoch, and the saved OOF/test come from the final EMA weights. Test = mean of the 5 fold models.
  - The cosine decay goes to 0 over a fixed number of steps (batches per epoch are constant).
  - Saved predictions never depend on an OOF epoch choice: they are the EMA weights or the snapshot mean, and `best_e` is only logged.
  - OOF is assembled from zero-filled held-out rows.

## Ideas
1. **(Highest value, no new code)** Fine-tune the v12 members with fta: `FTARGS="--decay cos --ema 0.99 --ema_start 0.5 --epochs 4"` (NNMATH-1). Expected ≈ +0.001 on Muon models (nnx).
2. Fix the aux normalisation (NNMATH-2 + NNMATH-3): one line, `aux.sum() / max_feeds`, so short/long feeds and exploration batches get the intended weight. Test as one paired 10-epoch arm.
3. Compute dense features in fp32 at inference (NNMATH-4). Free consistency fix: re-predict only, gain ≈ 0.
4. Retrieval over *distinct* history items (NNMATH-5). 26% of the top-16 slots are repeats and 41% of history events are repeats. Use fp32 S and a recency tie-break, and aggregate each item's repeats into the value (count, event-type mix, newest recency). That frees about 4 slots per candidate for new evidence. The same item-level dedup could feed a "times interacted" input to target attention.
5. Group user-0 feeds as singletons in the fine-tune folds (NNMATH-6). This keeps OOF and test treatment symmetric for 4.6% of feeds.
6. Short feeds and the domain token (NNMATH-8): in training, give M < 50 feeds the global prior p ≈ 0.057, or a learned "unknown" domain, instead of a hard production token. This removes the unseen (short feed, random-like token) combination at inference. It needs a paired run; do not pick the inference token on val.
7. Pass `it_hist_*_luo` only through the in-model within-feed rank/z (NNMATH-9; `XT_RELONLY` already supports it). Accept only if paired train-only val improves; private test has the least future in these counts.
8. Position is informative even in random-order feeds: holdout CTR is 0.021 at pos < 5 vs 0.0095 at pos ≥ 100, i.e. examination bias. With `--no_pos`, the train stage treats low-position negatives in random-like feeds as true negatives. A PAL position tower applied only to random-like and cooldown batches (where position is independent of relevance), dropped at inference, could reduce label noise at no inference cost.
9. Optimizer ablations (NNMATH-7, low priority): input projections on AdamW instead of Muon, and no weight decay on the embedding tables.
