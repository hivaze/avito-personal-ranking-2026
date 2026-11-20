# How the work was run, and what automation could change

In brief:

- One interactive session of Claude Code ran the nine days. The person on the team sent about 110 short requests and questions, wrote no code and uploaded the files.
- It was not an automated search: no unattended loop, no population of candidates, no second model and no scheduled review of the process. Uploads were by hand, as the platform requires.
- Published harnesses hold an example of the first three pieces. In the list checked on 19 and 20 November 2026, results in live competitions are few: one system with a gold medal, on a language-model task; one first place from a workflow guided by a person; and one paper that reports a top 1% placing.
- A stronger setup here would climb on paired differences on validation and the holdout, with a noise probe in every generation, and start with the fair runs of five ideas that were closed without one.
- More search does not lower the measurement floor: public cannot separate near copies below 0.0011 NDCG. The one search over ranking pipelines in that list found the gains of its first campaign to be almost entirely noise in its own score.

Notation follows [04-validation.md](04-validation.md#notation-and-files): scores are NDCG, "±" is one standard error (SE) of a paired per-feed difference, and a / b is validation / holdout. Holdout, near copy, try file and other terms are in the [glossary](glossary.md).

A harness on this page is the program around a language model that proposes a change, runs it, reads a score and decides what to keep. The glossary uses the word for a training directory of the lab.

## How the work was run

The tool was Claude Code, versions 2.1.303 (11 November) to 2.1.325 (19 November), running Claude Opus 5.5 at the xhigh reasoning effort in ultracode mode.

One lead session, a single long conversation with Claude Code, ran for the nine days. Its context window, the text a model can hold at once, filled and was summarised 21 times. Continuity came from written records: the status log, the decision ledger and a small file of standing instructions.

The standing instructions were short: keep the GPUs busy, write the rule before the run, record everything. Under them the lead session chose experiments, read the results and wrote the records. It wrote and launched workflows, scripts that start several agents in parallel and collect their reports. Ultracode mode allows this, and the status log names about 50. When a request was finished it stopped and waited for the next message.

The person on the team sent about 110 chat messages with a direction or a question, with a median length of 22 words. About 50 more only asked for status or passed a public score back. The person wrote no code, ran no experiment and analysed no result. The platform takes uploads only from a person, 5 per platform day (the day rolls over at 21:00 UTC), so the person uploaded the files and passed the scores back.

Who did what:

| Task | The person | The lead session | Agents inside a workflow |
|---|---|---|---|
| Chose the experiments | gave a few directions: tuned CatBoost, larger language models, stacking, pre-training on histories, final files trained on all data, try files | chose the rest under the standing instructions | searched for ideas when a workflow asked for it |
| Wrote the code | none | wrote the workflows and some read-out scripts | built the code inside workflows |
| Ran and read the experiments | none | launched the workflows and runs, read the results, wrote the records | ran training and evaluation and reported back |
| Checked the results | none | re-checked some reads itself | reviewer and refuter agents inside the same workflow |
| Looked back over the process | two short messages that changed a rule: not to reject an idea on the holdout alone (15 November); the rules are too restrictive, build try files (17 November) | ran audits after messages from the person (15 and 19 November) | carried out the audit of 15 November; nothing ran on a schedule |
| Uploaded files | all 40 scored files, at most 5 per platform day, and passed the scores back | prepared the files | built some files and ran the [pre-submission gate](04-validation.md#the-pre-submission-gate) |
| Chose the final picks | chose both ([11-final-picks.md](11-final-picks.md)) | set out the options in the decision ledger | none |

The two messages of 15 and 17 November changed which rule held and which files were built. The first re-opened three ideas about the popularity inputs ([04-validation.md](04-validation.md#rejecting-on-the-holdout-alone)). The second led to the try files, and both picks began as one. FDX16P20:ps, built from an idea that had failed its rule, scored 0.39526061 on public, the team's best, and is pick 2. The effect of both messages on the score is inside the noise of the board.

The scale: 40 scored files, 312 catalogued ideas, 151.6 GPU-hours for pick 1 from raw data plus 29.7 for the pseudo-feed students of pick 2, and rank 9 of 541 on the public board of 2026-11-19, 03:34 UTC ([README](../README.md#results)). The work ran on a few NVIDIA data-centre GPUs, whose model and number are not given here, and its total GPU time was not recorded.

![Two box-and-arrow diagrams, one above the other. Top, how this entry was run: a loop from the person, who sends a short request (about 110 in nine days), to the lead session (one session, its context summarised 21 times, a status log and a decision ledger as memory), to about 50 multi-agent workflows, to experiments on GPUs, to results and options written down, and back to the person, who uploads up to 5 files a day and passes the public score back. A marker in the person's box says that the loop waits there for the next message. Bottom, what an automated harness would add: a closed loop of five steps inside a frame, propose (checked against the catalogue of closed ideas), run (with a noise probe), evaluate (on validation and the holdout, under a rule written first), keep or drop, and archive, from which the next round starts without a message. Outside the frame stand the person, who sets the goal and the budget and uploads the files, a second model as critic, and a scheduled look back over the ledger.](figures/work-loop.svg)

The upper panel is the loop of this entry, which passes through the person and waits there for the next message; the lower panel is the closed loop proposed [below](#what-a-stronger-setup-would-do-here), with the person, a critic and a scheduled look back outside it.

## What the setup lacked

Four pieces were absent. The tables [below](#harnesses-that-exist-today) hold a published example of the first three, and the agents that rewrite themselves come nearest to the fourth. The table pairs each piece with a dated incident from the records. Whether the piece would have caught it is a judgement that no experiment backs.

| Missing piece | An incident it might have caught | What the incident cost |
|---|---|---|
| A loop that proposes, runs and selects experiments on its own schedule, day and night, without a message from the person | 18 to 19 November: the queue of GPU work was finished on 18 November at 05:18 UTC, and the next screen was launched on 19 November at 05:36 UTC. The status log records no training run in between. | about 24 hours in which no training ran |
| A population of candidate programs evolved against an evaluator, as in OpenEvolve, an open-source implementation of AlphaEvolve | 11 November: distillation, the training of a ranker toward the averaged scores of a pool, was closed after a student of one frozen 2-epoch teacher read +0.00064 on validation (z 1.6). A population that keeps the best candidate of each cell, as MAP-Elites does, could have kept this one for more runs. | three days: re-opened on 14 November, it became the largest late gain, +0.001032 on public |
| A second model from another vendor, for example Codex, taking part or reviewing | 19 November: a re-read of two closed arms against an average reference counted the arms' share of the seed luck as a gain (+0.00070 and +0.00045 on validation). The re-read came out of a workflow of four analysts and four verifiers; no model of another vendor took part. | a few hours in which two closed arms read as gains, and 8 more runs on fresh seeds; at 8 seeds the arms read +0.00006 and -0.00001 |
| A scheduled review of the process itself. Reviewer and refuter agents checked results inside a workflow. Two audits looked back over days of decisions (15 and 19 November), each after a message from the person; none ran on a schedule. | 11 to 15 November: the holdout vetoed popularity-light changes, which move weight to members that see one exposure count in place of the 8 popularity inputs. Public then rewarded them (-0.00063 on the holdout, +0.000523 on public). The veto ended after a message from the person. | four days; the audit of 15 November found 31 closures decided by a negative holdout and re-opened three ideas |

The same review would have met the strict rule sooner. Ideas with a positive mean that had failed it were built as try files only after the message of 17 November, and one of them became pick 2 (+0.00037 above FD16P20 on public, 0.8 paired SE). Uploads stayed by hand because the platform takes them only from a person, and no slip is on record there.

One slip fits none of the four. On 19 November the lead session proposed a cross-fitted teacher as untested. Such a teacher scores each feed with models that never saw its user, and it had been tested and closed on 14 November ([distillation](../experiments/distillation/README.md#the-cross-fitted-teacher)). No run was launched, so the cost is not measurable.

## Harnesses that exist today

The tables list published harnesses by kind. The list was checked against its sources on 19 and 20 November 2026: a second agent fetched the page named as the source of each entry, and the tables keep only what the checker read there. The working file, with the entries that failed the check, is `lab/reports/HARNESS_SURVEY_1120.md`. In them "±" is the error the source prints.

### Evolutionary search over programs

| System | What it does | Reported result | Source |
|---|---|---|---|
| AlphaEvolve (Google Cloud) | A code optimisation and discovery agent. The user gives a seed program, marks the segments that may change and supplies an evaluator that returns scalar metrics. The service returns mutated candidates and takes their scores back. The page describes the evaluator as deterministic. | Generally available on Google Cloud (post of 10 July 2026). Customer reports: forecast error (WMAPE) cut by over 5% against a production forecast in 200 iterations; 5 to 10% higher prediction accuracy across use cases; doubled throughput, with nearly 6,000 candidate programs over three weeks. | [cloud.google.com](https://cloud.google.com/blog/products/ai-machine-learning/alphaevolve-is-available-for-everyone) |
| OpenEvolve (open-source project) | An open-source implementation of AlphaEvolve. Language models mutate programs, which evolve against an evaluator the user supplies, with MAP-Elites (a method that keeps the best candidate in each cell of a grid of behaviours) and island populations (sub-populations that evolve apart). | "2-3x speedups on real hardware" for MLX Metal kernels and a state-of-the-art circle packing (n=26). No model-training or competition result is given. | [github.com](https://github.com/algorithmicsuperintelligence/openevolve) |
| EvoRank (University of Maryland and an independent researcher) | An open evolutionary loop guided by a language model, built on the AdaEvolve search controller of SkyDiscover. It discovers full learning-to-rank pipelines (features, models, losses, ensembles) on Expedia ICDM 2013 data, then re-scores the winners on 60k held-out queries. | Three runs each converged within 50 iterations, at about ten dollars each. Held-out NDCG@10 on 59,902 queries: 0.4316, 0.4322 and 0.4352 against 0.4296 for an Optuna-tuned LambdaMART. A late Kaggle submission of the full-scale pipeline: 0.5196 private NDCG@38, 20th of 340 (30 August 2026). | [arxiv.org](https://arxiv.org/abs/2609.22196), [full text](https://arxiv.org/html/2609.22196v1), [code](https://github.com/shabazpatel/evorank) |
| TabDAG (FusionBrain Lab) | An evolutionary (MAP-Elites) search over typed graphs of feature-engineering steps. A language model proposes graph changes that are checked before evaluation. The fitness is 3-fold cross-validation on the development split, and each cell's best program is frozen before the test evaluation. | Over ten datasets and three model families, TabDAG and TabDAG combined with OpenFE have the two best mean ranks among the compared methods. Pairwise rank differences between feature-engineering methods do not survive multiple-testing correction. A zero-shot probe (best of 16 samples) recovers a median 79% of the searched gain, which the paper calls a diagnostic and not a test result (17 November 2026). | [arxiv.org](https://arxiv.org/abs/2611.22828) |

### Agents that do ML engineering end to end

| System | What it does | Reported result | Source |
|---|---|---|---|
| MLE-bench (OpenAI), a benchmark | A collection of 75 Kaggle competitions for testing ML-engineering agents, scored by the share of competitions with any medal. The recommended run time is 24 hours. | Top entry of the main leaderboard, Famou-Agent 2.0: 64.44 ± 1.18% on all 75 (23 February 2026). An AIDE entry scored 17.12 ± 0.61% on 8 October 2024. An update of 24 April 2026 says no new submissions are being taken. | [github.com](https://github.com/openai/mle-bench) |
| AIDE (Weco AI) | A tree search over scripts: each Python script is a node, and a language model drafts, debugs or improves nodes, guided by the metric. | The README says only that MLE-bench found AIDE wins "4 × more medals" than the best linear agent. | [github.com](https://github.com/WecoAI/aideml) |
| MLEvolve (InternScience on GitHub) | An open-source system of several agents that solves Kaggle-style ML tasks end to end, with a graph search (Progressive Monte Carlo Graph Search), a memory of past experience and adaptive code-generation modes. | 65.3 ± 0.8% medal rate on the full MLE-bench (75 tasks) at a 12-hour budget, over 3 seeds, by its own README (entry of 1 June 2026). The arXiv abstract gives no number. | [arxiv.org](https://arxiv.org/abs/2606.06473), [code](https://github.com/InternScience/MLEvolve) |
| AIRA_2, on the AIRA-dojo framework | AIRA-dojo is a framework for research agents in which a solver is a search policy plus operators. AIRA_2 adds an asynchronous pool of workers on several GPUs, a "Hidden Consistent Evaluation" protocol for a reliable evaluation signal, and agents that scope their actions and debug interactively. | For one variant of AIRA_2, a mean percentile rank of 81.5% at 24 hours and 83.1% at 72 hours on MLE-bench-30, against 72.7% for the strongest baseline (13 April 2026). | [arxiv.org](https://arxiv.org/abs/2603.26499), [code of AIRA-dojo](https://github.com/facebookresearch/aira-dojo) |
| CoMind (research paper) | A system of several agents for ML engineering that uses outside knowledge from a simulated Kaggle research community and develops several solutions at once. MLE-Live is its live evaluation framework. | A 36% medal rate on 75 past Kaggle competitions within MLE-Live. In eight live, ongoing competitions it outperforms 92.6% of human competitors on average, with the top 5% on three official leaderboards and the top 1% on one (27 February 2026). | [arxiv.org](https://arxiv.org/abs/2506.20640), [code](https://github.com/comind-ml/CoMind) |
| AIRA3 (Meta) | An autonomous research system with no central controller. Many long-running agents work in isolated environments and coordinate through a forum for hypotheses and findings and a shared filesystem for solutions. | 8th of about 4,000 teams, with a gold medal, in a live Kaggle competition run by NVIDIA to fine-tune a 30B language model. Entered in June; the thread is dated 5 September, and the copy that was read shows no year. | [threadreaderapp.com](https://threadreaderapp.com/thread/2096271545589190927.html) |
| Agent workflow guided by a person (NVIDIA) | A person guided language-model agents in a human-in-the-loop workflow that follows a playbook: exploratory analysis, baselines, feature engineering, then hill climbing and stacking. | First place in the March 2026 Kaggle Playground competition on telecom churn prediction. The agents generated over 600,000 lines of code and ran 850 experiments; the winning solution is a four-level stack of 150 models selected from 850 (23 April 2026). | [developer.nvidia.com](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/) |

### Loops that run experiments unattended

| System | What it does | Reported result | Source |
|---|---|---|---|
| autoresearch (open-source repository) | An agent edits one file, `train.py`, trains for a fixed 5 minutes, checks whether the validation bits per byte improved, keeps or discards the change, and repeats. The person edits `program.md`, the instructions of the agent. | "approx 12 experiments/hour and approx 100 experiments while you sleep" on a single GPU. The page gives no other result. | [github.com](https://github.com/karpathy/autoresearch/blob/master/README.md) |
| MLE Agents (Meta Platforms) | Agents with engineers on the loop propose, code, train and evaluate changes to a ranker of search query suggestions. Each iteration trains nominally seven candidates plus a variance probe in parallel. A candidate is accepted only if it beats its probe by at least 0.0005 NE, the unit of its objective. Engineers control the objective, the direction and the final review. | 1,097 completed candidate runs over 131 iterations, of which 325 were accepted (29.6%). The best model gave +0.23% ± 0.15 search session volume in an A/B test of about one week (95% interval). Submitted on 18 November 2026. | [arxiv.org](https://arxiv.org/html/2611.23910) |
| Automated Alignment Researchers (Anthropic) | Nine copies of a language model, given sandboxes and tools, proposed, tested and analysed ideas on one research problem, weak-to-strong supervision, on their own for five days. | A final PGR, the share of a performance gap that is recovered, of 0.97; two human researchers working seven days had recovered 0.23 (14 April 2026). As of August 2026 a prototype index of the same organisation says its model "leads" 26% of its AI R&D work and is fully autonomous on no measured subset. | [study](https://www.anthropic.com/research/automated-alignment-researchers), [index](https://www.anthropic.com/institute/measuring-pace-of-ai-development) |

### Long-run modes of coding agents and review by a second vendor's model

| System | What it does | Reported result | Source |
|---|---|---|---|
| Claude Code, `/goal` and `/loop` (Anthropic) | `/goal` sets a completion condition, and after every turn a separate small, fast model judges whether it is met. `/loop` re-runs a prompt on a time interval. A research post of 23 March 2026 describes a recipe for long runs: a progress file kept as memory, a test oracle (reference code or tests), git commits and a loop that restarts the agent. | none in the checked text | [documentation](https://code.claude.com/docs/en/goal), [research post](https://www.anthropic.com/research/long-running-claude) |
| Codex, `/goal` (OpenAI) | `/goal` keeps Codex working across turns toward one objective with a verifiable stopping condition. It can run for many hours unattended and stops when it is confident the condition is met. The goal can be paused, resumed, cleared and edited. | none in the checked text | [learn.chatgpt.com](https://learn.chatgpt.com/use-cases/follow-goals) |
| Codex plugin for Claude Code (the openai organisation on GitHub) | A plugin for Claude Code. It provides `/codex:review` (read-only), `/codex:adversarial-review` (a steerable challenge of design and assumptions) and commands to delegate or hand off work. An optional review gate, a Stop hook, runs a Codex review of the response. | The page reports no result. | [github.com](https://github.com/openai/codex-plugin-cc) |

The lead session of this entry did not run under `/goal` or `/loop` with a research goal; it worked request by request.

### Agents that rewrite themselves

| System | What it does | Reported result | Source |
|---|---|---|---|
| Darwin Gödel Machine (research paper) | It rewrites its own code step by step, validates each change on coding benchmarks, and keeps an archive of the coding agents it has generated, from which new variants are sampled. | SWE-bench from 20.0% to 50.0% and Polyglot from 14.2% to 30.7% (29 May 2025). | [arxiv.org](https://arxiv.org/abs/2505.22954) |
| AIDE² (Weco AI) | An outer loop that rewrites the code of the inner-loop agent, AIDE, and keeps a rewrite only if it scores better. | 100 consecutive outer-loop steps with no human intervention in eight days of wall-clock time, and seven successive improved versions of AIDE. About nine in ten proposed changes were rejected; run-to-run noise was about 0.02 to 0.045 (14 July 2026). | [weco.ai](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement) |

### What these reports do not show

- MLE-bench, the source of most medal rates above, is a benchmark collection of 75 Kaggle competitions. It has taken no new submissions since a note of 24 April 2026, and the 65.3% is from the system's own README.
- One system in the list has a gold medal in a live competition: AIRA3, 8th of about 4,000 teams in a competition to fine-tune a 30B language model. The source is a social-media thread, read through a copy.
- CoMind reports, in its own paper, eight live competitions with the top 5% on three leaderboards and the top 1% on one. The checked text does not say what kind of competitions they were.
- The first place in the Playground competition came from a workflow that its own post calls human-in-the-loop.
- EvoRank, the one search over ranking pipelines, ran on the data of a 2013 competition, and its full-scale pipeline placed 20th of 340 as a late submission. Its paper says that re-scoring the winners on held-out data "showed the gains were almost entirely fitness noise" in its first campaign.
- The loop on a production ranker keeps engineers in control of the objective, the direction and the final review. It accepts a change only if it beats a variance probe trained in the same iteration, and its paper says the best model was found well before the end and later iterations added nothing.

Four entries guard against a noisy score: MLE Agents with a variance probe, AIRA_2 with an evaluation protocol, EvoRank with held-out re-scoring and TabDAG by freezing each cell's best program before the test evaluation. EvoRank also proposes a check to run before any search, in which one hand-built candidate must beat the noise floor of the fitness score by a set margin.

## What a stronger setup would do here

This is a design for this competition, where one 8-seed read on validation has an error of about 0.0005 NDCG. It is a proposal; none of it ran.

1. The fitness is the paired per-feed difference between a candidate and the reference, on validation and on the holdout, under a rule written before the run. The rule of 14 November can serve: at 8 seeds, both differences at or above zero and a pooled z of at least 2 ([04-validation.md](04-validation.md#decision-statistics)). For changes to the popularity inputs the holdout has no vote. The public score is never the fitness.

2. The search keeps a population of recipes, each one training configuration, and derives new candidates from the better ones. In the last screen one 10-epoch student run took 72 to 80 minutes on one GPU, so one GPU gives about 18 runs a day. A generation of 16 runs holds three candidates and the reference at 4 seeds each. A candidate with a positive read gets 4 fresh seeds.

3. Every generation carries a noise probe: the reference recipe re-run on the same fresh seeds as the candidates. Each candidate is read against it seed by seed, and the spread of the probes over generations measures the noise. On 19 November the fixed reference seeds sat about 0.0006 above the mean of the 70 possible 4-seed pools on validation, and a re-read against that mean counted the arms' share of that luck as a gain ([04-validation.md](04-validation.md#seeds-shared-by-arm-and-reference)).

4. The catalogue of 312 ideas ([07-what-failed.md](07-what-failed.md)) becomes a database with one row per idea and tested form, holding the kind of test (a fair run, a proxy or paper) and the read. Every proposal is matched against it before a run is planned. The row of the cross-fitted teacher, closed on 14 November, would have stopped the proposal of 19 November.

5. A second model, from another vendor, is the critic. It reads each plan before the run and each result before it enters the records, and argues against it. In this entry reviewer and refuter agents ran on the same model as the author. The page of the review plugin in the table reports no result, so this element is a bet.

6. A scheduled look back reads the decision ledger once a day in a fresh context and answers fixed questions: which rule closed the most ideas, whether a public read has contradicted it, and which closures rest on a single run. A public read contradicted the holdout veto on 13 November, two days before a message ended it. Distillation stayed closed from 11 to 14 November on one 2-epoch read.

7. Try files go out every night, within the 5 uploads per platform day that the person still makes. The harness builds them from candidates with a positive validation mean, runs the pre-submission gate and writes down the band of public scores in which nothing changes. Public scores tell how well validation and the holdout forecast the board. The picks would be decided on validation; in this entry both picks were taken after their public scores ([11-final-picks.md](11-final-picks.md)).

8. The first queue is the five ideas of [12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed). Each was closed on paper, by a proxy or in a flawed form, and gets one fair run of 8 seeds, read at the last checkpoint against the reference on the same seeds. Where the idea changes the training target, arm and reference both use fresh seeds.

   The cheapest is function matching: students trained on random variants of each feed with a consistent teacher target. Next-event training of the history encoder has the highest ceiling and takes several days to build.

By the lead session's judgement of 19 November, which no experiment backs, the chance that at least one of the five is worth +0.001 NDCG for a final file is about 0.25. None is expected to close the whole 0.003 to the leader.

## The limits such a setup would still meet

These findings of the records bound what any harness could measure on this board ([04-validation.md](04-validation.md#smallest-detectable-effect)).

- The 17 scored files of the final family span 0.39436 to 0.39526 on public (sd 0.00025), inside the paired SE of 0.0002 to 0.0006 between two of them ([04-validation.md](04-validation.md#near-copies-on-the-public-board)).
- Public cannot separate near copies below 0.0011 at 80% power. Validation needs 0.00086 and private 0.00073.
- A loop that uploads 5 files a day and keeps the best public score would mostly select noise. On 18 November pick 1 was already the best of 14 near copies, 0.00041 above their mean.
- More runs do not lower this floor. It comes from the click draw on 14,423 validation and 8,562 public feeds.

A harness therefore has three ways forward: climb on validation and the holdout with rules written before the run, build a larger evaluation set (for example by holding out more of the 164,118 random-order train feeds), or find effects large enough to see. EvoRank reports the second for its own search: enlarging the validation side of its fitness fold from 1.6k to 6k queries halved the noise, "the only intervention that moved it".

More search can buy breadth. One fair run of 8 seeds costs about 10 to 33 GPU-hours, and the build of an idea costs more, several days for next-event training. The ideas of the first queue were closed on paper before either was spent. A harness that runs day and night can afford the runs.

## One person's view of such competitions

This section is the opinion of the person on the team and not a finding. The README gives it in full ([My view on competitions like this](../README.md#my-view-on-competitions-like-this)).

In short: such competitions come with debatable data and a public board too small to rank the top places, and this entry reached rank 9 of 541 in nine days without an automated search. With the setups described above, the outcome will more and more show who runs the better harness, on more compute, for longer.

What keeps its value is what a reader can reuse: the reading of the data, the validation method and the negative results. That is why this repository documents them.
