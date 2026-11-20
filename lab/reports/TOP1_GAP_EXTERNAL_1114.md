# TOP-1 GAP, EXTERNAL VIEW (2026-11-14)

Scope: web research + one CPU probe. No GPU, no training scripts touched, no submission files. Read first: COVERAGE_MAP (all 297 lines), EXTERNAL_SCAN_1113 (§0, §3, §5, §7), EXTERNAL_TOP1_1113 §2, TRICKS_SWEEP_1113 headings, CURRENT_STATUS §1-6 and the 11-14 KD log.

Board: #1 0.398256, #2 0.395115, us 0.394486 (W8 wide KD students), #3 0.393778. Gap to #1 = 0.00377 public, i.e. 2.1-3.1 cross-team paired SE (0.0012-0.0018).

## 0. TL;DR

1. **Nothing public exists about this competition beyond the organiser page.** No write-up, repo, Habr post or indexed chat summary; the participant chat is an invite-only Telegram link. The page confirms what we already use (shuffled val/test, IDCG@30, public = first ~30% by time, two baselines).
2. **No external source gives a lever with an honest P(>= +0.002 public) above ~0.06.** The earlier scans already covered every winning recipe I could find; the write-ups not in their source list (VK-LSVD 2025, RecSys Challenge 2026, TAAC 2025, Trendyol 2025, WSDM Cup 2025) are different tasks or carry no transferable trick.
3. **The one thing that changed since 11-13 is our own KD result** (member +0.0026 val, +0.0017 public at pool level). It shows the ranker is label-noise limited, and it re-prices a family of levers that were closed or dropped *before* KD existed. Seven of the twelve levers below are in that family; they are the only ones with a published mechanism of the right kind.
4. Facts found in the code that matter for the list: `--kd_T` has never been varied (default 1.0 in every arm of `E_kd_predeclared.txt`); the teacher is a fixed per-candidate logit file scored on the original full feed, so no KD arm has used augmented candidate sets or unlabelled feeds.
5. CPU probe run for lever 3: a full-window leave-one-out item click rate on train has within-feed AUC 0.548 with 77% of rows covered at >= 5 other impressions. That is a moderately predictive train-only feature, the regime where privileged-feature distillation is reported to help.
6. Sum of expected pool gains over the whole table is about +0.0008 public. It does not close 0.0038. Most plausible reading of #1: a better-denoised model family plus a pool, not a hidden channel.

## 1. Ranked levers

Line references "l.N" are COVERAGE_MAP lines unless another file is named. P1 / P2 = my probability of >= +0.001 / >= +0.002 public at pool level. All are speculative unless marked "measured". Sources marked [A] were opened as abstract only, [S] search snippet only; the rest were opened and read.

| # | lever | source | mechanism | why it could reach +0.002 | why COVERAGE_MAP does not cover it | cheapest falsification | leak safety | P1 / P2 |
|---|---|---|---|---|---|---|---|---|
| 1 | **FM-KD: function-matching distillation on resampled candidate sets, with a longer schedule** | Beyer et al., arXiv 2106.05237 [A; method details from memory]; SDR, ar5iv 2109.15285 | Student matches the teacher on random sub-feeds (later: cross-feed candidate swaps) instead of on the one stored list. No fixed one-hot target to memorise, so extra epochs and width stop being harmful. | Our epoch (15/20/30) and width closures all failed through cooldown memorisation of click labels; width already turned positive under KD (d192 students). If the same holds for schedule length, the 10-epoch ceiling moves. | "SLS / multi-sample sub-list / :D sub-list" (l.168) were dropped without a run, pre-KD, with click labels (a sub-list loses positives). "kw_e20" under KD used the fixed stored list. | Inference-only probe: re-score 20K holdout feeds on random 50% sub-feeds with 4 teacher members. If within-sub-feed Spearman to the stored logits >= 0.97, the loader can sub-sample and renormalise stored logits at zero cost. Then one 2-seed kd05w192 pair with / without sub-feed sampling; kill at member < +0.0005 val or holdout < 0. | Train rows only; inputs unchanged at test. | 0.15 / 0.06 |
| 2 | **Transductive noisy-student KD: teacher soft labels on test (and val) feeds inside training** | WSDM Cup 2025 review, nebius.com/blog/posts/chatbot-arena-competition-review (winner = fold teachers distilled into students; pseudo-labelling listed as a common technique); CSRec, ar5iv 2311.02446 | 43K shuffled, in-period feeds enter the loss with ensemble soft targets and no click labels. They carry the test covariates (luo0 0.43-0.46 vs 0.13 on train). | The student is fitted where it is scored; the cold-item region is 3x under-represented in train. | "pseudo-labels (test)" (l.102) = hard labels in the old FT recipe; "KD / test pseudo-labels in the cooldown" (l.240) is RE-OPEN, never run; the reason given for FT-PSEUDO (l.285) concerned history-derived rows, not teacher-labelled real feeds. | CD harness, 4 roots: m02 cooldown + 2-5% test feeds with teacher targets (test feeds only, so val stays an untouched judge). STOP rule as for other cooldown arms. | Uses test features, never test labels; same construction for public and private rows; no later snapshot involved. Needs a line in the solution report. | 0.10 / 0.03 |
| 3 | **PFD-ITEM: teacher with train-only cross-feed item label aggregates** | Yang et al., NeurIPS 2022, arXiv 2209.08754 [A] | Teacher gets leave-one-out item click rate / COEC over the full train window (other users' labels, both time directions). Student never sees it and must explain item quality from content inputs, which transfers to cold test items. | Pools the same item's labels across feeds, which a same-input teacher cannot do. PFD reports gains over self-distillation; a too-predictive privileged feature hurts (this matches our hindsight closure, l.297, which was a label copy). | Item CTR / COEC / Bradley-Terry were tested as *inputs or residuals on val* (l.140, 234, 263), where they are sparse and decay. Never as a teacher-only feature. | **Measured (CPU):** LOO-CTR within-feed AUC 0.548 (COEC 0.544) on 40K train feeds; coverage 93% / 77% / 50% at >= 1 / 5 / 20 other impressions. Next CPU step: nested residual of LOO-COEC on the stored teacher logits over the 30K holdout; go only if >= +0.003. | Privileged feature exists only in the teacher on train rows. Exclude the same user's other feeds from the aggregate to stay inside the later-snapshot rule. | 0.08 / 0.03 |
| 4 | **Disjoint-teacher students (pool diversity under KD)** | CSRec data-level teachers (ar5iv 2311.02446: sub-sample ratio 0.8, two teachers beat one, more adds little) | Each student distils from a different teacher subset (by family or by user-hash data split). | Student-student corr is 0.978-0.990 vs 0.949 for base members, and "C3 + students = 0". The pool gain of KD is capped by that correlation, not by member strength. | All KD arms share one teacher file per arm (mean11, mean25, w8g1, kdmix). kd03 was "more diverse but weaker", which is a weight change, not a teacher split. | CPU, zero training: correlation of the existing kd05 (mean11) and kd05t25 students vs same-teacher pairs. If cross-teacher corr is not lower by >= 0.01, close. | Train-only. | 0.10 / 0.02 |
| 5 | **Teacher temperature / score transform** | SDR, ar5iv 2109.15285 (teacher score transformation is one of two named success factors; softmax transform underperformed affine; alpha 0.5) | kd_T < 1 sharpens a target whose median max prob is 0.08; an affine or top-k truncated teacher changes which pairs carry gradient. | Small alone; it multiplies every KD member. | `--kd_T` default 1.0 in all arms. | One 2-seed pair at T 0.5 and T 2 on the kd05w192 base. | Train-only. | 0.10 / 0.01 |
| 6 | **Soft-target fine-tune on val** | same as 5; WSDM Cup 2025 fold-teacher pattern | FT loss = (1-w) CE(y) + w KL(cross-fitted pool prediction). The 14.4K val feeds are where one-hot label noise is largest relative to data. | FT is worth only +0.0011 member / ~0 pool today; in-sample FT gains of +0.0095 show capacity to absorb val, so the limiter is noise. | FT retune, bagging, LP-FT, SAM, refit are closed (l.97-103, 139); none changed the target. | FT runs are minutes: 4 members, w 0.5, teacher = OOF pool prediction from other folds only. Read at pool level. | Teacher for a val row must come from folds that did not train on it. | 0.08 / 0.01 |
| 7 | **Noise-aware KD weight per feed** | CSRec ("confident" soft labels); EXTERNAL_SCAN N2 fact (41% of production M>=50 negatives sit below the last positive) | kd_w higher where click labels are least reliable (long production feeds, negatives below the last click), lower on exploration feeds. | Targets the specific noise source; N2 itself was never run. | N2 is UNTESTED-EXTERNAL as a logit offset; no feed-dependent kd_w exists. | CPU: per-slice agreement of teacher top-1 with clicks vs M and domain on the holdout; proceed only if the optimum w differs by slice. | Train-only. | 0.06 / 0.01 |
| 8 | Val-adapted teacher (FT pool scores the train feeds) | KDCRec / KDFCR label-based distillation [S] | Train targets carry the shuffled-feed label relation (popularity slopes are ~5x compressed on production feeds). | Moves in-distribution signal into 3.1M feeds instead of a 4-epoch FT. | OOF teacher (kx05) is a different thing (cross-fit over train users). | Cheap pre-check already argues against: main_prod vs main_rl differ by +0.0001 (l.157). Run only as a by-product. | Val OOF judge is contaminated unless teachers are fold-specific. | 0.04 / 0.01 |
| 9 | Meta-learned debiasing weights with exploration feeds as the meta set | AutoDebias, arXiv 2105.04170 [S] | Learns per-(pos, M, domain) weights and imputation instead of a fixed IPS / TRUNC form. | Published +11% NDCG@5 on Yahoo!R3 [S, unverified]. | IPS proxy, DW3, rho70, TRUNC are fixed-form. | None on CPU. Four fixed-form failures set the prior. | Train-only. | 0.03 / 0.01 |
| 10 | Teacher built from more families at equal weight per family | CSRec model-level; Zhu et al., arXiv 2011.04106 [S] | Family-balanced teacher (GRU, transformer, pop1 view, d192). | t25 gave +0.0007 member over mean11. | Partly covered (t25, kdmix); family balance not isolated. | Reuse existing dumps; one pair. | Train-only. | 0.05 / 0.005 |
| 11 | Two-generation chain with a *changed* student each time | SDR (no chain reported); our gen-2 closure | Gen 2 failed with the same student class; a chain only pays if the student gains capacity or data (levers 1-2). | Only on top of 1 or 2. | CURRENT_STATUS l.731 closed the plain form. | Free once 1 or 2 pass. | Train-only. | 0.03 / 0.005 |
| 12 | Team merge with a decorrelated top-5 team | ods.ai rules page | Cross-team corr is far below our 0.99 within-pool corr. | The only action with a frontier-law gain of +0.001-0.003. | GLOBAL_RECAP §6.4 (owner decision; deadline Dec 8). | Exchange val predictions only. | Rules-compliant; irreversible. | owner's call |

Order of work if GPUs are free: 4 (CPU, free) and 3 (CPU step) first; then 5 and 1 as pairs on the kd05w192 base; 2 in the CD harness; 6 on idle minutes.

## 2. Competition write-ups: decisive tricks (new to our source list)

| competition | decisive element | transferable? |
|---|---|---|
| WSDM Cup 2025 / LMSYS (nebius review) | fold teachers (70B) distilled by KL into small students, LoRA weights averaged | yes, pattern = levers 2, 4, 6 |
| VK RecSys Challenge LSVD 2025 (habr.com/ru/post/986440, a participant post; no winner write-up found) | user-side ranking per clip; content embedding mean + CF | no (retrieval task) |
| RecSys Challenge 2026 Music-CRS [S] | retrieval fusion + LambdaRank rerank, LLM text | no (text task) |
| Tencent TAAC 2025 (arXiv 2604.04976 [A]) | generative retrieval; findings not in the abstract | no |
| Trendyol / TEKNOFEST 2025 [S] | no write-up found | - |
| KDD Cup 2020 Debiasing 1st [S; already in EXTERNAL_SCAN] | popularity-weighted loss + post-hoc lift of rare items for a rare-item metric | no (our metric has no rare-item half) |

## 3. Hidden-structure tells: status here

| tell class (classic case) | status |
|---|---|
| stored row / array order (Expedia, KKBox) | audited, random (AUDIT_1112 l.43) |
| id / index / embedding-table order as a clock (ShareChat'23) | audited, null; item ids carry no time (l.278) |
| counters updated after the event (SNA'19) | f5 forward echo: leak-class, +0.0004, coverage < 5% |
| same-user later feeds (history snapshot) | real on val (+0.0045) but the 59 public feeds score nothing; banned |
| test-period item statistics, co-candidacy, duplicates | used (LUO) or null |
| timestamp reconstruction | f4 gives exact posting time; event-time lower bound exists for 59% of feeds, median 268 h, null (CODES_EXPLOIT l.34) |
| sample_submission order | key set checked; order differs from flat order but was never tested as a signal. One 2-minute CPU check (NDCG of row index within feed) would close it; P < 0.01 |

I found no unexamined channel with a prior above 0.01.

## 4. Small in-distribution set + large shifted set

Every method with competition evidence is already adopted or closed here: train-then-fine-tune (adopted), in-distribution replay late in training (m02, adopted), adversarial / density weighting (closed, ESS 2,880), stacking on val (closed, over-fits), val replay in the cooldown (closed). The uniform-data literature adds two forms not run: teacher-side use of the in-distribution data (lever 8) and learned weights (lever 9); AutoDebias itself reports that a teacher trained on the small uniform set alone falls to the biased baseline at a 1% uniform ratio [S], which matches our exploration-only model (0.382).

## 5. Most plausible explanations of the #1 gap

| hypothesis | P | change vs EXTERNAL_TOP1_1113 | reason |
|---|---|---|---|
| H2 better modelling: a denoising / distillation / diversity chain we have only started, or a stronger family | 0.55 | +0.05 | Our own single lever was worth +0.0017 public in one day, so a 0.004 modelling gap is no longer implausible; #2 is only 0.0006 ahead of us. |
| H7 organiser- or Avito-affiliated entry with extra inputs (event times, price, seller) | 0.10 | -0.02 | Entry count went 1 -> 3 with an unchanged score, weak evidence either way. |
| H3 heavier or smarter val-label use | 0.08 | -0.02 | No new protocol found externally. |
| H8 information channel we missed | 0.07 | -0.03 | §3: nothing left unaudited above P 0.01. |
| Public-specific fit or luck (weekend-only public, 8.6K feeds) | 0.08 | new split-out | Gap is 2.1-3.1 SE; a 3-entry row has little selection bias, but a weekend-favoured model would shrink on private. |
| H1 merge / H5 probing / H4 external data / H6 format | 0.12 total | = | Nothing new. |

## 6. Caveats

- Lever probabilities are judgement, not measurement. Only lever 3 has a number behind it, and that number is a marginal AUC, not a residual over the model.
- Beyer et al. and the PFD paper were read as abstracts only; the mechanism descriptions rely on my prior knowledge of those papers.
- I could not read the participant Telegram chat. If the owner has access, searching it for "distill", "pseudo", "leak", "pos" is the cheapest remaining information source.
- CPU probe script: /tmp/scratchpad/top1gap/loo.py (8 threads, reads data/flat/train only).

## 7. Sources opened

- https://ods.ai/competitions/avitotechrecsys2026
- https://ar5iv.labs.arxiv.org/html/2109.15285 (SDR / Born Again Neural Rankers)
- https://ar5iv.labs.arxiv.org/html/2311.02446 (CSRec, confident soft labels)
- https://arxiv.org/abs/2209.08754 (PFD in learning-to-rank, abstract)
- https://arxiv.org/abs/2106.05237 (patient and consistent teacher, abstract)
- https://arxiv.org/abs/2408.14678 (KD for online ranking, abstract; no usable detail)
- https://arxiv.org/abs/2604.04976 (TAAC 2025, abstract)
- https://nebius.com/blog/posts/chatbot-arena-competition-review
- https://habr.com/ru/post/986440
- Snippet only: arXiv 2105.04170 (AutoDebias), KDFCR / KDCRec (SIGIR 2020, TKDE 2023), arXiv 2011.04106, KDD Cup 2020 Debiasing 1st summary, RecSys Challenge 2026 model cards, TEKNOFEST 2025 results.
