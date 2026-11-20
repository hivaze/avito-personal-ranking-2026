# Modern training tricks, tested in paired groups

In brief:

- On 2026-11-11 recent techniques from language-model training and tabular deep learning were each tested on the neural ranker against one shared baseline (2 epochs, seeds 23 and 24).
- Adopted: Muon with a warmup-stable-decay schedule (+0.00362 ± 0.00075 on validation), retrieval of similar history items with a gate on attention (+0.00273 ± 0.00079) and a cosine weight-average fine-tune. Both main gains held on a fresh seed.
- A recheck of this harness found the holdout standard error about 2.3 times too small, which removed four reported wins. A later study of the main ranker did not reproduce the factor.
- On 19 November mixture of experts in current designs and a deeper student were tested on the final distilled recipe. All three arms failed at 4 seeds, and the two re-run at 8 seeds read +0.00006 and -0.00001 on validation.

## Why it was tried

On 2026-11-11 the ranker was a transformer over the candidates and the user's history ([neural-ranker-v1](../neural-ranker-v1/)). It was trained with AdamW, cosine decay and a separate second stage on random-order feeds, the train feeds displayed in random order like every validation and test feed.

The baseline looked limited by optimisation. Validation NDCG was still rising at the last step, and width, depth and head count had been neutral.

Four short research notes (`lab/reports/nnx_research_*.md`) gave a prior for each candidate, and most candidates cost one flag and one run.

## What ran

Three groups of experiments shared one GPU:

- attention (`attn_model.py`)
- inputs and heads (`feat_model.py`)
- the optimiser and schedule (`optim_train.py`)

Muon is an optimiser for the hidden weight matrices; AdamW handles the rest. The warmup-stable-decay schedule warms up, holds the learning rate constant, then lowers it in a cooldown. The cooldown mix is the share of cooldown batches drawn from random-order train feeds.

Every arm (one tested change) ran the full 2-epoch recipe for seeds 23 and 24 against the same baseline, whose mean was 0.38170 on validation and 0.30033 on the holdout. Every flag defaulted to the baseline code path, and the baseline weights had bit-identical initialisation.

A run took 38 to 76 minutes on a GPU shared by up to six such processes, a fine-tune about 8 minutes.

Comparisons are per-feed paired differences, pooled over seeds, on validation (14,423 feeds) and on the holdout of 30,000 random-order train feeds. The [glossary](../../docs/glossary.md) defines NDCG, the paired standard error (SE) and the other terms.

A separate check then:

- recomputed all of about 55 comparisons from the saved predictions, and all reproduced to five decimals
- measured noise with same-recipe seed pairs
- trained a fresh seed (25) of each adopted change

## Results

Train-only rankers (not fine-tuned on validation) except in the fine-tune row, 2 seeds unless stated. A "±" is one paired SE. The rows come from `results/*_results.tsv` and `lab/reports/nnx_VERDICT.md`.

| Change | Validation difference | Holdout difference | Outcome |
|---|---|---|---|
| Muon on hidden matrices + warmup-stable-decay schedule, 30% random-order cooldown mix, no second stage (against baseline + second stage) | +0.00362 ± 0.00075 (3 seeds +0.00306 ± 0.00067) | +0.00338 (3 seeds +0.00331) | adopted |
| Muon alone | +0.00201 ± 0.00060 | +0.00295 | part of the above |
| second stage after Muon + schedule | -0.00121 ± 0.00048 | -0.00147 | stage dropped |
| retrieval of the 16 history items most similar to each candidate | +0.00186 ± 0.00077 | +0.00326 | adopted |
| retrieval + per-head sigmoid gate on attention outputs | +0.00273 ± 0.00079 (3 seeds +0.00278 ± 0.00074) | +0.00389 (3 seeds +0.00382) | adopted |
| cosine + weight-average fine-tune on Muon checkpoints (out of fold) | +0.00112 ± 0.00038 | | adopted |
| forgetting bias | -0.00019 | -0.00014 | rejected |
| query-key norm with length-aware temperature | -0.00006 | -0.00010 | rejected |
| mixture of experts, 8 experts, top 2 | +0.00027 ± 0.00047 | +0.00053 | rejected |
| piecewise-linear embeddings of numeric inputs | -0.00043 ± 0.00060 | +0.00078 ± 0.00037 | rejected |
| propensity head | +0.00007 | +0.00014 | rejected |
| semantic-ID codes | +0.00038 | +0.00003 | rejected |
| smoothed history features | -0.00002 | -0.00021 | rejected |
| weight-sharing mini-ensemble (TabM, k = 4), 3 seeds | +0.00009 ± 0.00042 | +0.00032 | rejected, 1.45x run time |
| weight EMA (moving average of the weights), decay 0.9998 | -0.00057 ± 0.00026 | -0.00051 | rejected |
| self-distillation from one earlier model | +0.00064 ± 0.00041 | +0.00098 | not demonstrated |
| Muon learning rate 2e-3 against 1e-3, new schedule | -0.00097 ± 0.00058 | +0.00056 | kept 1e-3 |

"Out of fold" means each validation feed is scored by the fine-tune that did not train on its fold.

The fresh seed replicated both adopted changes on validation: retrieval + gate +0.00289 ± 0.00087, Muon + schedule +0.00194 ± 0.00089.

## What it taught

The per-feed SE of the holdout was too small. Two runs of one recipe with different seeds moved the holdout by about 0.001, at a reported z near 2.8. The run-to-run standard deviation was about 0.00085 against a per-feed SE of 0.00037, so holdout z-scores had to be read 2.3 times smaller. On validation the per-feed SE matched the seed noise.

With that correction, four reported wins fell below 2 SE: the gate alone, the gate over retrieval, self-distillation and learning rate 2e-3.

The factor rested on 3 pairs of runs. A later review of the main ranker, with 15 same-recipe seed pairs, found root-mean-square z of 0.83 on validation and 0.89 on the holdout, so later pages use the per-feed SE for both witnesses (`lab/reports/REVIEW_SYNTHESIS.md`).

The two gains were of different kinds. Retrieval gave the target attention information it lacked. The term for how a matched history item differs from the candidate carried it: without that term, the holdout read -0.00288 against full retrieval.

Muon changed how the content projection was used, raising its effective rank from 30 to 79. In round 7, moving the input projections from Muon to AdamW cost -0.00101 on validation (`lab/reports/R7_ARCH_HPARAM.md`).

The attention variants engaged as designed and left the score within noise, since they rearranged information the model already used.

Piecewise-linear embeddings rose on the holdout and fell on validation. They learned step functions of production traffic that do not hold on random-order feeds, and the holdout rise was within run noise.

The check also warned about the cooldown mix. A 30% mix is about 2 passes over the 134,118-feed pool of random-order train feeds at 2 epochs and about 9 at 10 epochs, a regime that had not been tested.

The warning came true. All six 10-epoch members trained with the 30% mix collapsed in the cooldown: validation fell from about 0.391 to 0.364 (`lab/reports/REVIEW_SYNTHESIS.md`, problem P1). The mix went to 10%, then 2% ([rounds-5-7](../rounds-5-7/), and [failure 3](../../docs/07-what-failed.md#3-a-cooldown-dose-tuned-at-2-epochs-and-used-at-10) in the docs).

Distillation, closed here with one teacher, returned three days later in ensemble form ([distillation](../distillation/)).

## Mixture of experts re-tested on distilled students (19 November)

The test above was too small to settle the question, and in it the last block's router sent almost every token to one expert. On 19 November three arms were built on the final student recipe (width 192, 5 blocks, 2.12M parameters) and run with 4 seeds each against the stored students.

The designs came from recent language models. A routed expert is a small feed-forward block to which a learned router sends each token, here the top 2 of 8. A shared expert is a dense block that every token passes through. DeepSeek-V4 and Kimi K3 balance the load with a per-expert bias on the router scores instead of an extra loss. SwitchHead routes the value and output projections of attention.

| Arm | Parameters | 4-pool validation | 4-pool holdout | Outcome |
|---|---|---|---|---|
| A: 8 routed feed-forward experts of width 48, top 2, dense layer kept as a shared expert, balancing bias set by quantile | 2.71M (2.27M active) | -0.00086 ± 0.00057 | +0.00020 ± 0.00031 | closed |
| B: routed value and output projections in the 2 cross-attention blocks, 4 experts, top 2 | 2.72M (2.43M active) | -0.00144 ± 0.00057 | +0.00029 ± 0.00032 | closed |
| C: 7 blocks instead of 5, feed-forward ×1 | 2.27M | -0.00069 ± 0.00058 | +0.00040 ± 0.00034 | closed |

Router entropy stayed at 0.78 or higher, so no expert took the whole load, yet arm A used its experts little. Their output was 0.07 to 0.21 of the dense layer's, and switching them off at inference cost -0.00010 ± 0.00014. In the last block 2 of 8 experts received no tokens on 3 of 4 seeds.

The screen read checkpoints chosen on the holdout, which picked steps 85,000 to 90,000 for 4 of the 12 runs. Students trained on all data keep the last step. At the last checkpoint, paired by seed, the arms read +0.00009 (A), -0.00093 (B) and -0.00016 (C) on validation.

An analysis then re-read A and C against the mean of the 70 possible 4-seed pools of the 8 stored references, because the screen's reference pool (seeds 951 to 954) ranks 3rd of them on validation. That gave +0.00070 ± 0.00046 and +0.00045 ± 0.00047.

Seeds 955 to 958 of A and C settled it under a rule written first. The rule compared 8-pools at the last checkpoint with the 8 references and asked for +0.0003 on validation.

| Arm | 8-pool validation | 8-pool holdout | Outcome |
|---|---|---|---|
| A | +0.00006 ± 0.00047 | -0.00004 ± 0.00026 | closed |
| C | -0.00001 ± 0.00048 | +0.00027 ± 0.00026 | closed |

The re-read had counted the seeds' luck twice. On validation, seeds 951 to 954 read above seeds 955 to 958 by +0.00113 for the reference and by +0.00082 and +0.00085 for A and C, so arms that share the teacher and the seeds share the luck, and the seed-paired screen had already cancelled it ([04-validation.md](../../docs/04-validation.md#seeds-shared-by-arm-and-reference)).

The gain from pooling 8 seeds matched the reference's for both arms, so the experts did not make the seeds more alike. Arm B fit its own training feeds better (+0.0025 NDCG in sample) and moved away from the teacher, so its extra capacity went into the training clicks. Neither experts nor depth changed the distilled student.

Rule and reads: `lab/nn_x/w12/moe/RULE_predeclared.txt`, `lab/nn_x/w12/moe/logs/` (`read_ext8.txt`, `seedluck.txt`, `paired_last.txt`, `poolgain8.txt`), the analysis in `lab/nn_x/w12/moe/why/`.

## Files

`code/`:

- `optim_train.py`: trainer of the optimiser group, with switches for Muon, the schedule, the cooldown mix, the auxiliary anneal, EMA and distillation.
- `optim_finetune.py`: fine-tune with cosine decay, weight average and linear probe.
- `optim_paired.py`: paired read-out.
- `attn_model.py`: model with retrieval, gate, forgetting bias, query-key norm and mixture of experts.
- `attn_judge.py`: read-out and verdict rule of the attention group.
- `feat_model.py`: model with the five rejected input-side changes.
- `verify_recheck.py`: recomputation of every comparison, with the SE calibrated on same-recipe seed pairs (null calibration).
- `verify_seed25.py`: fresh-seed replication.

`results/`:

- `optim_results.tsv`, `attn_results.tsv`, `feat_results.tsv`: one row per comparison.
- `verify_recheck.txt`: recomputation and null pairs.
- `optim_env_base_flags.txt`: the shared baseline flags.

### Raw material and records

- `lab/archive/nnx/optim/`: the run JSON per arm of the optimiser group. The other groups and the check are in `lab/nn_x/attn/`, `lab/nn_x/feat/` and `lab/nn_x/verify/`.
- The per-candidate predictions behind the comparisons (216 files, 3.9 GB) are not in this repository.
- Reports under `lab/reports/`: `nnx_design.md`, `nnx_exp_optim.md`, `nnx_exp_attn.md`, `nnx_exp_feat.md`, `nnx_VERDICT.md` (section 4 covers the port into the main ranker) and `nnx_research_*.md`.
