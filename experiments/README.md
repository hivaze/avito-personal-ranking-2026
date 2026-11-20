# Research lines

In brief: 25 folders, one per line of work from the nine days (2026-11-11 to 2026-11-19), each readable without the lab tree. The final pipeline is not here. It lives in [`avito_ranker/`](../avito_ranker/), and [docs/03-solution.md](../docs/03-solution.md) explains it.

## What a folder holds

- `README.md`: the write-up of the line, under the same headings in every folder: In brief, Why it was tried, What ran, Results, What it taught, Files.
- `code/`: six to eight scripts copied from the lab. The only change is a header that says how the script was run and where the original lives.
- `results/`: the read-outs that carry the numbers.

The scripts are kept as the record of what ran. They spell absolute paths of the original machine and read arrays and predictions that are not published, so they do not run from this tree as copied. `lab/README.md` says what a run would need.

The READMEs use the lab's code names next to their plain meaning, and [docs/glossary.md](../docs/glossary.md) lists them all. The full catalogue of ideas, one row each, is in [docs/07-what-failed.md](../docs/07-what-failed.md). The folders below are the lines worth reading in depth. [docs/13-how-the-work-was-run.md](../docs/13-how-the-work-was-run.md) tells how the nine days were run.

A newcomer can start with five folders in this order: [data-understanding](data-understanding/) for the data, [neural-ranker-v1](neural-ranker-v1/) for the model, [distillation](distillation/) for the largest late gain, [popularity-staleness](popularity-staleness/) for the gap between fresh and stale inputs, and [validation-and-leaks](validation-and-leaks/) for what each score can and cannot show.

## Understanding the data and measuring correctly

- [data-understanding](data-understanding/): how train, validation and test differ, and what the embeddings and the collaborative signals carry.
- [exploration-detection](exploration-detection/): which train feeds were shown in random order, when the data has no flag for it.
- [codes-and-embeddings](codes-and-embeddings/): what the anonymised event, source and category codes and the item vectors are, and whether knowing it helps.
- [validation-and-leaks](validation-and-leaks/): which signals are leaks, and how far validation, the holdout and the public board can be trusted when two files differ by less than 0.001 NDCG.
- [error-analysis](error-analysis/): where the best pool loses NDCG and whether any signal separates its misses.

## The model and how it is trained

- [neural-ranker-v1](neural-ranker-v1/): whether a set ranker over the whole train split beats boosted trees, and which part carries the gain.
- [optimiser-and-modern-tricks](optimiser-and-modern-tricks/): which recent training techniques help, each against one shared baseline, and, on 19 November, mixture of experts and depth for the distilled students.
- [rounds-5-7](rounds-5-7/): screens of the loss, schedule, inference and blocks once the recipe was settled.
- [training-waves](training-waves/): changes to the cooldown, regularisation, length, batch and structure of the click-label ranker (waves 5 to 11), read on pools, and the cooldown re-test of the students on 17 November.
- [feature-families](feature-families/): which pre-computed candidate columns add information and can be built the same way for train and test.
- [popularity-views](popularity-views/): rankers that see little or no popularity, as extra pool members, and the weight of the popularity-light students in the final mix.
- [popularity-staleness](popularity-staleness/): stale popularity inputs on test, and the holdout that disagreed with the public board.
- [pretraining](pretraining/): a sequence encoder pretrained on the user histories, as inputs, as weights or as a ranker.
- [fine-tune](fine-tune/): how to learn from the 14,423 validation feeds and what the fine-tuned half is worth.
- [distillation](distillation/): students trained on half clicks, half the pool's averaged scores; the teacher variants, the student recipe arms of 14 to 16 November, and the pseudo-feed students of pick 2.
- [full-data-retrains](full-data-retrains/): retraining the final members on all train feeds, and how to accept them without a holdout.

## Combining models

- [hard-label-pools](hard-label-pools/): which click-label rankers go into a pool and how their scores are combined.
- [pool-size-and-growth](pool-size-and-growth/): what a pool gains from more seeds, other students or learned weights, with 16, 32 or 48 members.
- [mixing-and-segments](mixing-and-segments/): feed-dependent mixing weights, weak user segments, exposure counts from other feeds.

## Ideas that did not make it

- [gbdt-and-stacking](gbdt-and-stacking/): boosted trees as rankers, as a second stage, or as pool members.
- [catboost-round-2](catboost-round-2/): CatBoost with teacher targets and native ranking losses as a pool member.
- [llm-rankers](llm-rankers/): language-model trunks over projected vectors, and a general model ranking feeds directly.
- [trained-input-search](trained-input-search/): information the students do not use yet, trained in as a new input.
- [tricks-sweeps](tricks-sweeps/): the remaining small tricks, such as post-processing, pooling forms, regularisers and test-time augmentation.
- [external-solutions](external-solutions/): other competitions' solutions, outside data, and what could explain the leader's margin.

Five ideas that never had a fair training run, because they were closed on paper, by a proxy or in a flawed form, are listed in [docs/12-why-the-score-stopped.md](../docs/12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed).
