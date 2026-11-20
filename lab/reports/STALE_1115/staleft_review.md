# staleft_review: adversarial review of STALE-FT (KEY = staleft)

Written 2026-11-15 09:15-09:47Z by the review agent. Work directory `nn_x/w12/stale/staleft_review/`. CPU only; no GPU process; no process of anyone else touched. Conventions: +- = 1 SE paired per feed on the 14,423 val feeds unless a count is given; [M] measured by me with my own code, [D] read in a file (path), [I] inferred. "Study" = `reports/STALE_1115/staleft.md` and `nn_x/w12/stale/staleft/`.

## 0. Verdict

1. **CONFIRMED: STALE-FT fails the lead's rule on every read, and closing it is right. Do not spend the 6-9 GPU-process-hours of a 3-seed run.** Every number of the study's verdict reproduces to the last printed digit with independent code (own NDCG, own z, own SE) [M].
2. **"Missed +2 SE by 0.07 SE" overstates how close it came.** The pooled-W8 read (+0.00054 +- 0.00028, z 1.93) is one high draw. The per-feed delta averaged over ALL member subsets of the same 8 members reads, P = 1.0 final stale-val: +0.00017 +- 0.00014 (70 four-member pools), +0.00027 +- 0.00017 (28 six-member pools), +0.00020 +- 0.00020 (8 seven-member pools): 1.0 to 1.6 SE. Fresh val: -0.00006, +0.00004, +0.00010. Only 9% of the 70 possible four-member pilots pass the lead's rule; no seven-member pool does [M].
3. **No dose effect, and nothing at the private mix.** P = 1.0 minus P = 0.5 on the subset-averaged mix read: -0.00001 +- 0.00006, +0.00003 +- 0.00007, +0.00002 +- 0.00009. Stale-val at pmix-private: 0.00000 +- 0.00022, -0.00001 +- 0.00027, -0.00024 +- 0.00032. In feeds with cold share >= 45% (8% of val feeds; 21% of the public and 33% of the private weight [D study]) all ten reads are negative, -0.0001 to -0.0018, none at 2 SE [M].
4. **Leak and validity audit: clean.** Out-of-fold discipline intact, stale OOF written from the held-out fold only, test rows read real inputs, P = 0 executes no new statement in the training step, all 48 one-seed jobs differ only in member and dose. Two counterfactual columns rebuilt from raw by me: max abs diff 0 [M].
5. **Two side claims of the study need correcting.** (a) "The fine-tuned half of the popularity-light students is not popularity-light": true as a column list, false in behaviour: with its 18 popularity extras switched to stale that pool loses nothing (+0.00026 +- 0.00025), while the W8 fine-tuned pool loses -0.00173 +- 0.00053. The stale cost sits in the TRAIN-STAGE inputs, not in the fine-tune extras. (b) "Adds nothing once the popularity-light admixture is in the file" rests on an interaction of -0.00039 +- 0.00031; the P20-form FAIL stands, the mechanism is not shown [M].
6. **A flaw of the stale view the study did not name.** Four extras were kept fresh inside it; one (`it_hist_all_luo_gap`) marks every "newly cold" row exactly (1.000 of newly cold rows, 0.000 of always-cold rows). The P > 0 arms are trained and scored with that channel open; real stale rows do not have it. It can only inflate the stale-val gain, so the FAIL is conservative [M / I].
7. **GPU conduct: inside the grant.** Two processes at a time, every launch under the lock with 3 / 3 / 3 / 3 / 2 / 2 processes present. Two detached queue processes of the study still hold the 4th slot of GPUs N and 6 (PIDs 249332, 254220): 9 of 20 jobs left at 09:44Z, end about 10:20Z; no lane was waiting on those locks at 09:24Z [M].

## 1. Claim table

"My number" is from `nn_x/w12/stale/staleft_review/out/rederive.txt` unless another file is named.

| claim of the study | its number | my number [M] | agree |
|---|---|---|---|
| Pilot 4-pool, P = 0.5, final fresh / stale-val | -0.00021 +- 0.00020 / +0.00024 +- 0.00019 | same (z -1.07 / +1.24) | yes |
| Pilot 4-pool, P = 1.0, final fresh / stale-val | -0.00017 +- 0.00025 / +0.00042 +- 0.00024 | same (z -0.69 / +1.77) | yes |
| Pilot, P = 1.0, final stale-val at pmix-public / pmix-private | +0.00047 +- 0.00028 / +0.00048 +- 0.00033 | same | yes |
| Pilot, FT branch alone, P = 1.0, fresh / stale | -0.00028 +- 0.00037 / +0.00035 +- 0.00040 | same | yes |
| A/A (P = 0 run twice), final fresh / stale-val | -0.00003 +- 0.00003 / +0.00004 +- 0.00005 | same | yes |
| Replication 4-pool, P = 1.0, final fresh / stale-val | -0.00005 +- 0.00025 / +0.00024 +- 0.00028 | same | yes |
| Pooled W8, P = 1.0, final fresh / stale-val | +0.00022 +- 0.00026 / +0.00054 +- 0.00028 (z 1.93) | same for that pool; subset-averaged +0.00010 +- 0.00018 / +0.00020 +- 0.00020 at k = 7 (`out/poolsize2.txt`) | number yes; **"misses by 0.07 SE" no** (2.1) |
| Pooled W8, P = 1.0, stale-val at pmix-private | -0.00006 +- 0.00044 | same | yes |
| Member level, 8 members, P = 1.0, fresh / stale | -0.00033 +- 0.00020 / +0.00026 +- 0.00023 | same (z -1.62 / +1.13) | yes; a "clean trade" at 1.6 / 1.1 SE |
| "Replicates in both halves of W8" | stale-val +0.00042 and +0.00024 | true for that 4 + 4 split; three other complementary splits I tried give (+0.00011, -0.00013), (+0.00027, -0.00005), (-0.00008, -0.00026) (`out/extra_checks.txt`) | **no** |
| Dose: P = 1.0 beats P = 0.5 on pooled W8, mix read | +0.00024 +- 0.00013 | same on the 8-pool; subset-averaged -0.00001 +- 0.00006 (k = 4), +0.00003 +- 0.00007 (k = 6), +0.00002 +- 0.00009 (k = 7) | number yes; **claim no** |
| P = 0.5 pooled W8, final fresh / stale-val | -0.00004 +- 0.00019 / +0.00033 +- 0.00022 | same | yes |
| W8P20 form, P = 1.0, final fresh / stale-val | +0.00001 +- 0.00025 / +0.00015 +- 0.00024 (z 0.62) | same; stale-val at pmix-private -0.00036 +- 0.00038 | yes |
| W8P20 form, P = 0.5, final fresh / stale-val | -0.00016 +- 0.00019 / +0.00019 +- 0.00019 | same | yes |
| "Adds nothing once the popularity-light admixture is in the file" | +0.00015 (P20 form) against +0.00054 (W8 form) | interaction, P20-form delta minus W8-form delta: stale -0.00039 +- 0.00031 (z -1.28), fresh -0.00021 +- 0.00028 | FAIL yes; **mechanism not shown** |
| Stale inputs cost the control final / FT branch / train-only pool (4-pool) | -0.00175 +- 0.00050 / -0.00164 +- 0.00053 / -0.00156 +- 0.00050 | same | yes |
| Popularity-light admixture at final-file level: stale / stale M >= 50 / fresh | +0.00056 +- 0.00034 / +0.00111 +- 0.00035 / +0.00036 +- 0.00031 | same; stale at pmix-private +0.00036 +- 0.00051 | yes |
| Test shift, final file, top-1 changed: P = 1.0 / P = 0.5 / A/A | 4.4% / 2.4% / 0.07-0.09% | public, private: 4.46%, 4.41% / 2.39%, 2.37% / 0.07%, 0.09% | yes |
| Val at the public / private mix reads the fine-tune half (stored 3-seed W8) | +0.00066 +- 0.00054 / +0.00104 +- 0.00067 | same (z 1.22 / 1.56); plain val +0.00009 +- 0.00043 | yes; a direction, as the study says |
| 1-seed W8 file equals the stored 3-seed file | +0.00001 +- 0.00014 | same | level yes; a one-seed pool DELTA is 49-75% member-combination noise (2.1) |
| "Fine-tuned half of the popularity-light students is not popularity-light" | 7 of 8 excluded columns re-enter through the extras | column list confirmed [D `nn_x/w12/stale/staleft/logs/abl_e10_pop1kd05t25_s951_sft_p0_ft.log`]; behaviour: that pool's stale minus fresh = +0.00026 +- 0.00025 (`out/poplight_stale.txt`) | list yes; **conclusion no** (2.4) |
| Zero share of `it_hist_all_luo`: val 0.379, stale view 0.485, private 0.456, public 0.428 | as stated | 0.3794, 0.4850, 0.4558, 0.4276 (`out/gate_raw.txt`) | yes |
| Stale view farther from PUBLIC rows than fresh val on the it_hist family | dZ 0.051 -> 0.055, dM 0.085 -> 0.108 | 0.0510 -> 0.0546, 0.0845 -> 0.1082 | yes |
| Counterfactual extras reproduce exactly | max abs diff 0 | my rebuild FROM RAW of `both_it_hist_all_luo` (existing file) and `both_it_hist_et5_luo` (built by the study): max abs diff 0; 0 of 1,389,971 rows differ | yes |
| Identity run, W16 member, fold 0 | corr 0.999999; +0.000303 +- 0.000203 on 2,885 feeds | corr 0.999999; +0.000303 +- 0.000203; top-1 differs on 0.28% | yes (bar missed by the letter, as self-reported) |
| Identity run, popularity-light student | -0.00004 +- 0.00006 | -0.000036 +- 0.000057 | yes |
| Job time at 1 seed | 6-9 min, mean 7.7 | 36 finished jobs: 4.3-8.9 min, mean 7.4 [D queue logs] | yes |
| GPU: two processes, lock, never more than 4 per GPU | 3 / 3 / 3 / 3 / 2 / 2 processes at the six launches | launch logs agree; GPUs N and 6 at 4 processes at 09:16Z, one of them the study's (section 4) | yes |

## 2. Issues by severity

### 2.1 MAJOR for the reading, not for the verdict: one pool delta is a noisy draw; the subset average says about +0.0002, not +0.0005 [M] (`out/poolsize.txt`, `out/poolsize2.txt`, `out/extra_checks.txt`)

The rule's statistic is the paired NDCG delta of two final files built from one-seed fine-tune pools. I recomputed it for every subset of the 8 W8 members (final = 0.5 z(FT pool) + 0.5 z(TO pool) over the same subset), and I averaged the PER-FEED delta over all subsets of one size before taking its SE.

| pool size k (subsets) | P = 1.0 final fresh: subset-averaged +- SE | P = 1.0 final stale-val: subset-averaged +- SE | single-pool stale-val: SD over subsets; range | stale-val at pmix-private, subset-averaged | subsets passing the lead's rule | P = 0.5 fresh / stale-val, subset-averaged |
|---|---|---|---|---|---|---|
| 1 (8) | -0.00029 | +0.00013 | 0.00026; -0.00023 .. +0.00064 | +0.00003 | 0% | -0.00014 / +0.00012 |
| 2 (28) | -0.00011 | +0.00011 | 0.00027; -0.00046 .. +0.00051 | 0.00000 | 0% | -0.00004 / +0.00004 |
| 4 (70) | -0.00006 +- 0.00013 | **+0.00017 +- 0.00014** (z 1.19) | 0.00026; -0.00040 .. +0.00061 | 0.00000 +- 0.00022 | 9% | +0.00002 +- 0.00007 / +0.00011 +- 0.00009 |
| 6 (28) | +0.00004 +- 0.00016 | **+0.00027 +- 0.00017** (z 1.60) | 0.00027; -0.00024 .. +0.00098 | -0.00001 +- 0.00027 | 7% | +0.00006 +- 0.00009 / +0.00019 +- 0.00011 |
| 7 (8) | +0.00010 +- 0.00018 | **+0.00020 +- 0.00020** (z 1.01) | 0.00025; -0.00041 .. +0.00042 | -0.00024 +- 0.00032 | 0% | +0.00005 +- 0.00011 / +0.00022 +- 0.00014 |
| 8 (1) = the study's pooled read | +0.00022 +- 0.00026 | +0.00054 +- 0.00028 | - | -0.00006 +- 0.00044 | fails | -0.00004 +- 0.00019 / +0.00033 +- 0.00022 |

What this shows:
- **49-75% of the per-feed variance of a pool delta is member-combination noise** (0.74 at k = 4, 0.62 at k = 6, 0.49 at k = 7): it re-draws when one member changes. Dropping ONE member of eight moves the stale-val delta anywhere between -0.00041 and +0.00042. The per-feed SE of one pool read is honest for that one draw, but reads on other member subsets are nearly independent draws, and their average is the better estimator.
- The 8-pool value +0.00054 lies above all eight of its own seven-member pools. The stable estimate of STALE-FT on the stale view is **about +0.0002 at 1.0-1.6 SE** (in-distribution, an upper bound), **0 at the private mix**, 0 to +0.0001 on fresh val. By the lead's bar (+2 SE) it fails at every pool size.
- "P = 1.0 beats P = 0.5" is a property of the 8-pool draw. Subset-averaged, the two doses are equal within 0.00003.
- The study's sentence "the fresh-val loss disappears as the pool grows" IS supported (-0.00029, -0.00011, -0.00006, +0.00004, +0.00010).
- "It replicates in both halves" should read: the member-level trade (fresh -0.00033 +- 0.00020, stale +0.00026 +- 0.00023) has the same sign in both halves. Neither side is 2 SE.
- For the lead beyond this study: when a pool-level read of one-seed pools sits near +2 SE, compute the subset average before believing it. It costs CPU minutes and its SE is half as large at k = 4 (0.00014 against 0.00025).

### 2.2 MAJOR, supports the close: the gain is not where the private mix has its weight [M] (`out/extra_checks.txt`, `out/poolsize2.txt`)

Final-file delta against the control by feed cold share (fresh-val cold share of `eda/errors/segments/feed_table.parquet`), stale-val:

| read | cold share >= 45% (1,213 feeds) | cold share < 45% (13,210 feeds) |
|---|---|---|
| P = 1.0, W8 form, the 8-pool | -0.00140 +- 0.00116 | (three bins: +0.00053, +0.00063, +0.00089) |
| P = 1.0, W8P20 form, the 8-pool | -0.00179 +- 0.00110 | (three bins: +0.00042, +0.00038, +0.00022) |
| P = 1.0, subset-averaged, k = 4 / 6 / 7 | -0.00024 +- 0.00052 / -0.00050 +- 0.00065 / -0.00123 +- 0.00078 | +0.00021 +- 0.00015 / +0.00035 +- 0.00018 / +0.00033 +- 0.00021 |
| P = 0.5, W8 form / W8P20 form, the 8-pool | -0.00088 +- 0.00099 / -0.00108 +- 0.00094 | - |
| P = 0.5, subset-averaged, k = 4 / 6 / 7 | -0.00011 +- 0.00033 / -0.00038 +- 0.00041 / -0.00082 +- 0.00051 | +0.00013 +- 0.00010 / +0.00024 +- 0.00012 / +0.00031 +- 0.00014 |

No cell of the coldest bin reaches 2 SE and the rows are not independent, but all ten are negative, in the bin that carries a third of the private weight and that the study's own side finding names as the place where the fine-tune half earns its value. STALE-FT, as far as it does anything, helps moderately cold feeds under the synthetic view and costs on the coldest. The study's section 5 does say "the stale-val gain is NOT where the private mix puts its weight"; its returned verdict does not.

### 2.3 MEDIUM: the stale view leaves a channel open that real stale rows do not have [M / I] (`out/gate_raw.txt`)

The study's gate kept four extras fresh because they failed its magnitude bar (`it_hist_all_luo_gap`, `it_hist_et7 / et9 / et10_luo`) while their parent count is stale. Label-free consequences:
- `it_hist_all_luo_gap` equals fresh count minus fresh feed maximum exactly (max abs diff 0).
- Of the rows whose STALE count is 0 (48.5% of val rows), 21.8% are "newly cold" (fresh count > 0). The kept-fresh gap separates them from always-cold rows perfectly: gap > -feed max on 100.00% of newly cold and 0.00% of always-cold rows; with within-feed information only (gap > feed minimum gap) 99.97% against 0.00%. Pointwise the gap says nothing (correlation with log1p fresh count -0.02), so the channel needs within-feed context.
- `it_hist_et7 / et9 / et10_luo` > 0 on a row whose stale all-count is 0, a pattern no real row has: 3.4% / 0.9% / 3.9% of newly cold rows.
- Footprint in the predictions (label-free; 13,441 feeds holding both kinds of rows; mean per-feed z on newly cold rows minus always-cold rows, stale view, 8 W8 members): train-only +0.269, P = 0 fine-tune +0.309, P = 0.5 +0.316, P = 1.0 +0.322. P = 1.0 minus P = 0 = +0.0128 +- 0.0003; P = 0.5 minus P = 0 = +0.0070 +- 0.0002. The fresh view reads +0.407, so P = 1.0 recovers 13% of the fresh / stale gap.

I cannot separate "learned the open channel" from "learned legitimate proxies of warmth" without re-scoring the fine-tuned models on a consistent stale view, and the study saved no fine-tuned checkpoint. Both readings make the measured stale-val gain an upper bound, so the FAIL is conservative. It is also a reason not to read the absolute stale cost of the FINE-TUNED branch (-0.00164) as fully private-like.

### 2.4 MEDIUM: corrections to two "where the data disagree with the status files" items [M]

- **"The fine-tuned half of the popularity-light students is not popularity-light."** Column list: correct (the student's run switches 18 extras and 3 train-stage columns [D log]). Behaviour (`out/poplight_stale.txt`, stale view minus fresh view, standard fine-tune, 1 seed):

  | pool | fine-tuned pool | train-only pool |
  |---|---|---|
  | popularity-light d128 x4 | +0.00026 +- 0.00025 | +0.00007 +- 0.00011 |
  | W8 | -0.00173 +- 0.00053 | -0.00176 +- 0.00052 |

  Stale extras cost nothing: the popularity-light pool loses nothing with 18 of them stale, and the W8 fine-tuned pool (stale train-stage inputs AND stale extras) loses no more than the W8 train-only pool (stale train-stage inputs only, no extras at all). The stale cost sits in the 8 train-stage popularity / co-visitation columns. Nothing has to be rebuilt for the popularity-light fine-tunes. Caveat [D study, section 7]: the students' one popularity input `xp_imp_c50` stays fresh in this scenario, so their train-only insensitivity is partly by construction; the statement about the extras does not depend on that.
  This also explains the null [I]: STALE-FT has to re-train how the trunk reads 8 train-stage columns, with 4 epochs at lr 5e-5 on 11,500 feeds. Robustness to stale inputs is bought at the train stage (popularity-light students, stale-aware train arms), not in the fine-tune.
- **"Val under-reads the fine-tune half because of 3 fold-models against 15."** The study's alternative (val at the public mix reads +0.00066 +- 0.00054) reproduces, at 1.2 SE. Neither explanation is shown; the status-file sentence should be marked "one of two open explanations", not replaced.

### 2.5 MINOR

- The "decisive read" on the W8P20 form: z 0.62 is a FAIL, but it is one more single draw (2.1), and it is not evidence that the admixture absorbs the effect (interaction -0.00039 +- 0.00031). Both forms say "+0.0002 or less".
- The W8P20 stale-val read uses the popularity-light fine-tuned branch at P = 0 in both arms (the study says so).
- Identity bar missed by the letter (+0.000303 against 0.0003), self-reported. What I measured in support: the untrained (epoch-0) predictions of the P = 0, P = 0.5 and P = 1.0 runs of one member are bit-identical (max abs diff 0), so folds, start weights and inference are deterministic inside the study; the offset to the stored file belongs to the stored file's run, not to the new code.
- Rule 4 breach (three report drafts in the session scratchpad), self-reported. No project file outside the study's two locations carries its names (`find` on `preds/`, `nn_x/w5/preds`, `nn_x/abl/preds`, `subs/`; the only new "stale" files there are the lead's own `nn_x/w5/preds/stale/*`).

## 3. Leak and validity audit

| question | finding |
|---|---|
| Out-of-fold discipline | Intact [D `code/finetune_stale.py`]. Same `GroupKFold` on user with `--u0_singleton` as the original; the stale mix is applied only to batches drawn from the training folds (`src is va`, batches from `tr_f`); `oof_fin_stale += predict(ema, vs, ho_f)` scores the held-out fold only. Empirical [M]: the 96 val files of the 8 W8 members hold no constant or all-zero feed, all 1,389,971 rows finite. |
| Test rows | Always read the real test inputs: the path switch acts only for `split == "val"` and is set back to off before training [D]. |
| Labels in the stale inputs | None. The counterfactual columns are counts minus later counts (formula rebuilt by me from the raw history arrays: exact) and encoder scores with the item-ID part removed on newly cold rows. Extras are standardised with the fresh val statistics, as test rows are [D]. |
| P = 0 identical to the original | Code: no statement of the training step runs at P = 0 (`if a.stale_train > 0`); what does run in addition is the construction of the second view, an unused numpy generator, and two extra `predict` calls in eval mode under `no_grad`. The frozen copies of `data.py`, `model.py`, `blend.py`, `finetune.py` are byte-identical to `nn_x/w5` (md5 by me); `common.py` differs only in output paths. All 48 one-seed jobs carry the same recipe arguments. Not bit-reproducible against the stored file on GPU (2.5). |
| Stale val closer to test than fresh val on every switched column | Against PRIVATE rows: yes on the four columns I recomputed (`it_hist_all_luo`, `it_hist_et5_luo`, `seq_cv_pop`, `pre_id_contrib_v2`: dZ 0.080 -> 0.028, 0.016 -> 0.007, 0.063 -> 0.033, 0.054 -> 0.023). Against PUBLIC rows: NO for the it_hist family (0.0510 -> 0.0546; `it_hist_et5_luo` 0.0104 -> 0.0112), yes for `seq_cv_pop` and `pre_id_contrib_v2`. Two instances were switched against the study's own magnitude bar by its amendment 1 (declared before any GPU run). "Closer on every switched column" is therefore false as stated, and the study says so itself. |
| Is the stale-val gain circular | Partly. Training and scoring use the same synthetic transformation on different users, so the read shows that the model can adapt to THIS transformation, artefacts included (coverage beyond both test parts, magnitudes moved the wrong way, the open channel of 2.3). It does not show transfer to naturally stale rows. The fresh-val column bounds the cost if test rows behave like val. The label-free test shift (4.4% of top-1s, equal on public and private days) shows the change is real in size but not aimed at the staler days. A gain on stale-val is necessary, not sufficient; a FAIL on it is informative. |

## 4. GPU conduct [M / D]

- Launch logs `nn_x/w12/stale/staleft/logs/launch_{A,B,C,E,G,H}.log`: every queue process started after a blocking `flock` on `nn_x/w5/logs/locks/gpu<g>.lock`, with the process count and free memory checked under the lock (3 / 92 GB, 3 / 97 GB, 3 / 97 GB, 3 / 96 GB, 2 / [n] GB, 2 / [n] GB) and the lock held until the process was visible. The launcher is a faithful copy of the `ft_lane.sh` mechanism.
- Never more than two study processes: A (07:00-08:10Z) and B (07:10-08:13Z); C from 08:10:02Z after A ended 08:09:58Z; E from 08:13:07Z after B ended 08:13:04Z; G from 08:56:31Z after C ended 08:56:29Z; H from 09:01:11Z after E ended 09:01:10Z.
- `nvidia-smi` at 09:16Z (look only): GPUs N and 6 hold 4 compute processes each: two full-data runs, one `train_full2.py` run, and the study's queue (PID [n] with `CUDA_VISIBLE_DEVICES=N`, PID [n] with `CUDA_VISIBLE_DEVICES=N`, 12.4 GB each). No GPU above 4.
- Design note, not a breach: one queue process keeps its slot across all its jobs (one locked launch per queue, not per job). A lane of the lead cannot use that slot until the queue ends.
- Still running after the study returned: queues G and H, 11 of 20 jobs done at 09:44Z, end about 10:20Z. No holder and no waiter on the GPU N / GPU N locks at 09:24Z (`lslocks`, `/proc/locks`).
- No failure line in any queue log (36 jobs finished at 09:23Z). I could not verify the two early launchers the study says it stopped (PIDs 14117, 14150): no trace remains.

## 5. What the lead should do with STALE-FT

1. **Close it as a replacement of the fine-tune. No 3-seed run.** Record it as "about +0.0002 at 1.0-1.6 SE on an in-distribution synthetic witness, 0 at the private mix, negative in the coldest feeds, no dose effect", not as "missed by 0.07 SE".
2. **Do not re-open it on the W16 single-pool read.** If the two queues are left to finish, read them with the statistic I wrote before any W16 control existed (`nn_x/w12/stale/staleft_review/DECLARATION_SUGGESTED.txt`, 09:26Z and 09:44Z): the lead's fresh-val clause, the stale-val delta AVERAGED over the 16 leave-one-member-out pools at +2 SE, stale-val at pmix-private >= 0, and both 8-member halves positive. A pass would mean "build the 3-seed branch and read the P20 form", not "ship".
3. **The two slots.** The W16 read cannot change a file sent today. Keep the queues only while no lane of the lead needs GPU N or GPU N; otherwise stop PIDs 249332 and 254220 by exact PID (finished jobs keep their files).
4. **Do not build a "STALE-FT v2" with a better stale view.** 2.4 says the fine-tune is the wrong stage: the stale cost is carried by the train-stage inputs. What addresses it is already running at train stage, plus the popularity-light share of the file (+0.00056 +- 0.00034 at final-file level under stale inputs, M >= 50 +0.00111 +- 0.00035, reproduced).
5. **No public probe.** The test files for a W8-form or P20-form twin exist, but the expected effect is +0.0002 or less against a paired public SE of about 0.0004 [D study], and a clean pair needs a second submission for the one-seed control.
6. **Status files.** (a) Keep "3 fold-models against 15" as one of two open explanations of the fine-tune half's public value. (b) Do not adopt "the popularity-light fine-tuned half is not popularity-light" as a problem to fix. (c) Add the subset-average check of 2.1 to the evidence policy for pool-level reads of one-seed pools.

## 6. Method, commands, files

All from `<repo>` with `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12 PYTHONDONTWRITEBYTECODE=1`; at most two processes of mine at a time.

```
P=.venv/bin/python; D=nn_x/w12/stale/staleft_review
$P $D/rederive.py        # own NDCG (numpy lexsort, pessimistic ties), own z, own SE: pilot, replication, pooled W8, W8P20 form, A/A, test shift, FT-half worth -> out/rederive.txt
$P $D/gate_raw.py        # raw rebuild of 2 counterfactual columns, own gate of 4 columns, kept-fresh-extras facts and prediction footprint -> out/gate_raw.txt
$P $D/extra_checks.py    # OOF completeness, identity re-read, cold-share split, member-set sensitivity -> out/extra_checks.txt
$P $D/poplight_stale.py  # stale minus fresh of the popularity-light and W8 pools -> out/poplight_stale.txt
$P $D/poolsize.py        # every member subset by pool size: spread of the single-pool statistic -> out/poolsize.txt (9 min)
$P $D/poolsize2.py       # per-feed delta averaged over all subsets of size 4 / 6 / 7, with its SE, pmix and cold-share splits -> out/poolsize2.txt (3 min)
```

- NDCG: binary gain, pessimistic ties (within a tie the negatives rank first), float32 scores, feeds without a positive count as 1. z = per-feed population z. Final = 0.5 z(FT pool) + 0.5 z(TO pool). W8P20 form = per branch 0.8 z(W8) + 0.2 z(popularity-light x4). pmix weights from `src/pmix.py::weights()` (the project's); weighted mean and SE by my own formula (effective n).
- Files: `nn_x/w12/stale/staleft_review/{rederive,gate_raw,extra_checks,poplight_stale,poolsize,poolsize2}.py`, `out/*.txt`, `out/perfeed.npz`, `DECLARATION_SUGGESTED.txt`, `PROGRESS.md`; this report.
- One process of mine was stopped by exact PID after reading its command line: a shell wait-loop (PID 282451) that matched its own name.
- My own slip, corrected in place: the first block of `DECLARATION_SUGGESTED.txt` says the per-feed SE does not contain the member-set noise. `poolsize2.py` showed it does (49-75% of it); the second block corrects the sentence and the statistic, still before any W16 control existed.

## 7. What I did NOT check

- No GPU work: no fine-tune re-run, no re-scoring of a fine-tuned model on a consistent stale view (no fine-tuned checkpoint exists), so 2.3 is not resolved.
- The W16 extension: not read (its controls had not run). `DECLARATION_SUGGESTED.txt` is a reviewer's proposal; the lead's rule is the lead's.
- The gate for all 33 column instances: I recomputed 4 columns and rebuilt 2 counterfactual files from raw. The 13 other files the study built (within-feed transforms, the `pre_` family) were not rebuilt.
- `code/test_view_cpu.py` and `code/test_main_cpu.py` of the study were not re-run.
- The subset analysis uses the 8 W8 members with one fine-tune seed each; its subsets overlap, so the subset average estimates the expected effect for THESE members, and the "member-combination" share mixes member and fine-tune-seed noise.
- The W8P20 form was not put through the subset analysis.
- The cold-share bins use the fresh-val cold share; under the stale view every feed is colder.
- Whether the lagged holdout or the hz2 causal construction would read differently: not available for a fold fine-tune on val (the study says the same).
- CURRENT_STATUS, E_kd_predeclared, veto.md, witness.md and hz2.md were not re-read in full; I relied on the task text for the standing assumptions.
- No submission file, no `src/presub_gate.py` run, nothing sent anywhere.
