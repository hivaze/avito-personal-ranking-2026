# State of the art and winning solutions

In brief:

- Comparable competitions are won by gradient-boosted re-rankers over hand-built features, by transformers over user histories and, where the data allow, by counts taken from the test period.
- The outside ideas that paid here were the shape of Avito's own ranker, a fine-tune on the validation days, random-order traffic found inside train, co-visitation counts, optimiser and attention tricks, and distillation from an ensemble, whose first file scored +0.001032 on public over the best click-label file.
- Test-period counts, item-ID models, trees beside the network, extra capacity and language models did not transfer. Listings rarely repeat inside the scored period, many candidate items appear in no history, and the ranker is limited by labelled data.
- The gap to the top of the board is most likely better modelling or better use of the training data (P 0.64, a probability from the records of 18 November; see the [hypotheses](#hypotheses)). Apart from a leak that scores nothing on public, every channel inside the files probed on 18 November reads at most +0.00003 NDCG on top of pick 1 ([below](#what-the-top-of-this-board-may-use)).

Scores, splits and notation follow [04-validation.md](04-validation.md), and terms are in [glossary.md](glossary.md). The tables on this page use these conventions:

- Validation is the mean NDCG (normalised discounted cumulative gain, the competition metric) over the 14,423 validation feeds. The holdout is 30,000 random-order train feeds that train-only members never see. Public is the public leaderboard, 8,562 of the 28,595 test feeds.
- "a / b" is a paired per-feed difference on validation / holdout, ± one standard error (SE). z is a difference divided by its SE. Public scores are marked as public.
- A member is one trained ranker, and a pool is the equal-weight mean of its members' scores, standardised within each feed. A click-label member learns from clicks alone. A student learns from clicks and from the scores of an ensemble of earlier members (distillation).
- The cooldown is the last 20% of training, where the learning rate decays. "Out of fold" means read on validation feeds that the fine-tune fold did not train on. A residual is a new signal fitted with one weight on top of a finished pool, in folds split by user.

## The problem class

The task re-ranks a given list. A feed is one user's request with 5 to 300 candidate listings, the label is a binary engagement, and the score is NDCG over the full list, averaged over feeds. The data hold 3.15 million train feeds, 14,423 validation feeds and 28,595 test feeds.

Train is logged from production and is position-biased. On the Avito main page CatBoost picks the top 300 candidates, a neural model re-ranks them and a category blender sets the final order. Validation and test hold only feeds whose candidates were shuffled uniformly before display.

The closest relatives are impression-ranking competitions, where the candidates are given: RecSys Challenge 2024, VK RecSys Challenge 2024, Avito ML Cup 2024 and the Expedia hotel-search competition of 2013. Session and next-item competitions (OTTO, H&M, Amazon KDD Cup 2023, Avito ML Cup 2025) add a retrieval stage that this task does not have.

## What wins in comparable competitions

### Levers that recur

- Two stages, candidate generation and then a re-ranker. In retrieval tasks recall decides, and the H&M winners called their candidate generation "the key". Here the candidates are given, so only the second stage applies.
- Gradient-boosted decision trees (GBDT: LightGBM, CatBoost YetiRank, XGBoost) as the re-ranker, over hundreds of interaction features plus the scores of first-stage models. In OTTO the winner's neural model alone reached 0.515 weighted Recall@20, against 0.588 for the LGBMRanker built on top of it. Yandex reports the same for its ARGUS transformer, whose score is worth more as a GBDT feature than on its own ([below](#research-state-of-the-art)).
- Sequence transformers, where histories are long and data are large. VK RecSys 2024 second place used a pure transformer, pretrained and then fine-tuned, with no feature engineering. The RecSys 2024 winners' core model is a transformer, and RecSys 2025 places 2 and 4 are sequence encoders. Avito's own production re-ranker is a transformer ([below](#avitos-own-ranker)).
- Test-period and leak features, which decide some boards. The RecSys 2024 winner drops from AUC (area under the ROC curve) 0.8864 to 0.7699 without test-period impressions. For Expedia 2013 the fifth-place paper reports 0.52477 NDCG@38 with counts over train and test and 0.50099 without. A top-3 solution of Avito ML Cup 2024 used the within-day frequency of the scored day, and the TalkingData 2018 winner used next-click time deltas.
- Training on the newest labelled period from the test distribution. Retraining with the newest week gave each RecSys 2024 winning model +0.0097 to +0.0123 test AUC, and adding the dev day took the winner of MIND, a Microsoft news-recommendation competition, from AUC 0.707 to 0.7104.
- Ensembles. Averaging several models added +0.3% to +0.4% relative for the winners of the Yahoo learning-to-rank challenge, KDD Cup 2022 and MIND. Seed bagging is standard (a KKBox write-up, WSDM Cup 2018: "never submit a single NN"), and second-stage stackers are trained on the in-distribution period (RecSys 2024 first place, NVIDIA's third place in RecSys 2022).
- Distillation and soft pseudo-labels. The KDD Cup 2022 task 1 winner gained +0.0022 public NDCG (0.8960 to 0.8982) from self-distillation plus soft pseudo-labels on test. The WSDM Cup 2025 winner distilled fold teachers into students, and the RecSys 2024 winner used soft pseudo-labels on test impressions.
- Validation discipline. The second places of Santander 2016 and TalkingData 2018 trusted local cross-validation over a small public board.

### Recap of comparable competitions

Where a winner published nothing, the row says so and gives the best write-up found.

| Year | Competition | Task and metric | Winning approach | Sources |
|---|---|---|---|---|
| 2013 | Expedia hotel search (Kaggle, ICDM 2013) | Rank hotels per search; train mixes production and random-order impressions (flag `random_bool`). NDCG@38 | The winner's method was not read. The fifth-place paper reports counts over train and test, 0.52477 with them against 0.50099 without. | [arXiv 1311.7679](https://arxiv.org/abs/1311.7679) |
| 2022 | H&M Personalized Fashion Recommendations (Kaggle) | Twelve articles per customer for the next week. MAP@12 (mean average precision) | First place: candidates mostly recent popular items, a logistic-regression pre-ranker, then LightGBM and CatBoost classifiers on 340+ interaction features. Image and text features did not help. | Kaggle write-ups through the [Hugging Face (HF) mirror](https://huggingface.co/datasets/Abirate/all_write_ups) |
| 2022-23 | OTTO multi-objective recommender (Kaggle, Nov 2022 to Jan 2023) | Clicks, carts and orders per session. Weighted Recall@20 | First place: about 1,200 candidates per session from multi-hop co-visitation and a neural model, then LightGBM's LGBMRanker, with a mean of 9 rankers at 0.605 on the board. Removing the co-visitation features cost 0.0052, removing the neural scores 0.0053. | write-ups through the HF mirror; third-place code [Kaggle-OTTO-Comp](https://github.com/cdeotte/Kaggle-OTTO-Comp) |
| 2023 | Amazon KDD Cup 2023 (Amazon-M2) | Next product in multilingual sessions. MRR@100 (mean reciprocal rank) | NVIDIA-Merlin, first in all three tasks: candidates from co-visitation, text embeddings and session models, then XGBoost trained on 360 million rows (+0.002 MRR from the larger training set). | [poster](https://github.com/kddcup23/kddcup23.github.io/blob/main/posters/NVIDIA-Merlin-Poster-v2.pdf) |
| 2024 | RecSys Challenge 2024 (Ekstra Bladet news, EB-NeRD) | Rank the articles inside an impression. AUC, also MRR and nDCG@5 and @10 | Team ":D" (DeNA): a transformer plus LightGBM and CatBoost, a second-stage stacker, a retrain with the newest labelled week and test-period impression features. Ensemble test AUC 0.8924. | [organisers' paper, arXiv 2409.20483](https://arxiv.org/abs/2409.20483); [slides](https://speakerdeck.com/kfujikawa/enhancing-news-recommendation-with-transformers-and-ensemble-learning); [code](https://github.com/k-fujikawa/recsys-challenge-2024-1st-place) |
| 2024 | VK RecSys Challenge (VK Clips) | Like, ignore or dislike for given user-clip pairs. Per-user ROC-AUC over 3 levels | No first-place write-up found. Second place: a pure transformer, pretrained on (action, author, clip) events and then fine-tuned, "two submissions were enough". Third: CatBoost on about 250 click-rate, ALS (matrix factorisation) and similarity features. Fifth: a small multi-head network, single model 0.6732, seed average 0.6773. | [ods.ai/competitions/aivkchallenge](https://ods.ai/competitions/aivkchallenge) |
| 2024 | Avito ML Cup 2024 (ad click-through) | Rank ads per user. Per-user ROC-AUC | The best write-up found is a top-3 solution (second in its code, third in the Habr results post): CatBoost YetiRank over target means of every 1- to 3-way key combination, the within-day frequency of the scored day, a pairwise-aware target encoding and a BPR (Bayesian personalised ranking) network score. | [Habr results post 864532](https://habr.com/ru/articles/864532) |
| 2025 | Avito ML Cup 2025 (personal recommendations) | Item groups a user will contact. Recall@40 | The 2025 winners: EASE-DAN and RDLAE linear autoencoders and SASRec, a sequential transformer recommender, as scorers, then CatBoost YetiRank averaged over folds. | [ods.ai/competitions/avitotechmlchallenge2025](https://ods.ai/competitions/avitotechmlchallenge2025) |
| 2025 | RecSys Challenge 2025 (Synerise) | Universal user representations, scored on churn, category and product (stock-keeping unit, SKU) propensity plus hidden tasks. AUC-based aggregate | No first-place write-up found (team rec2). Second, ai_lab_recsys (Sber AI Lab): a GRU (gated recurrent unit) event-sequence autoencoder plus an ensemble. Fourth, a Yandex team: a 20-layer causal transformer with a DCN-v2 (deep and cross network) head and 300+ hand-built features, where the next-item objective mattered most. | [arXiv 2508.07748](https://arxiv.org/abs/2508.07748); [arXiv 2508.06970](https://arxiv.org/abs/2508.06970) |

Left out as not comparable: Amazon KDD Cup 2024 (shopping tasks for language models, won in all five tracks by NVIDIA with a 72-billion-parameter instruction model fine-tuned per track, arXiv 2408.04658), RecSys Challenge 2026 (conversational music recommendation) and VK RecSys Challenge 2025 "LSVD" (the top 100 users per cold clip, NDCG@100, winners' methods not found).

A web search on 2026-11-18 found no write-up of this competition, and neither did the search of 2026-11-14.

## Research state of the art

In feed ranking the research frontier is a transformer that reads the user's action sequence together with the candidates, trained on far more data than this task has.

- HSTU (Meta, "Actions Speak Louder than Words", ICML 2024, [arXiv 2402.17152](https://arxiv.org/abs/2402.17152)) reports up to 65.8% higher NDCG than baselines on synthetic and public data and +12.4% in an online A/B test, at 1.5 trillion parameters.
- ARGUS (Yandex, "Scaling Recommender Transformers to One Billion Parameters", [arXiv 2507.15994](https://arxiv.org/abs/2507.15994)) reports +2.26% listening time and +6.37% like likelihood online. In pairwise accuracy its score gives +2.32% as one feature of a GBDT ranker with about 1,000 features and -5.33% on its own, and pretraining roughly doubles the gain of fine-tuning alone (+1.17% to +2.32%).
- OneTrans (ByteDance, October 2025, [arXiv 2510.26104](https://arxiv.org/abs/2510.26104)) puts sequence and non-sequence features through one tokenizer and one transformer. The authors report +5.68% per-user gross merchandise value (GMV) online.
- OneRec (Kuaishou, arXiv 2506.13695) replaces the retrieve-then-rank cascade with one generative model. It was read only through a secondary summary.
- Two surveys cover the area: generative recommendation ([arXiv 2510.27157](https://arxiv.org/abs/2510.27157)) and recommenders in production ([arXiv 2509.06002](https://arxiv.org/abs/2509.06002)).

### Avito's own ranker

In October 2026 Avito described its main-page ranking on Habr ([article 1090584](https://habr.com/ru/companies/avito/articles/1090584/)). CatBoost passes its top 300 candidates to a neural re-ranker. The best model is one transformer over the user's history, filtered to target actions, and the feed's candidates. Candidates attend to the history and, with a mask, to each other (about +0.6%).

The output of the transformer feeds a parallel DCN (deep and cross network), the loss is ListMLE (the likelihood of the observed order) and the model has about 2.5 million parameters.

Offline it scored +7.2% NDCG on target actions over a CatBoost ranker with about 200 features. Online it added +4.3% users with contacts, +3.3% with target actions and +0.8% with a deal. CatBoost and network ranks are blended with equal, untuned weights. The ranker of this solution is of the same kind, with candidates that attend to the user's history and to each other, at 2.12 million parameters ([03-solution.md](03-solution.md#the-ranker)).

### Limits the literature sets

- The loss can matter as much as the architecture. SASRec trained with full softmax cross-entropy matches or beats BERT4Rec (Klenitskiy and Vasilev, RecSys 2023, [arXiv 2309.07602](https://arxiv.org/abs/2309.07602)).
- On Web30K, Yahoo and Istella a tuned neural ranker (DASALC) only reaches parity with LightGBM LambdaMART (Qin et al., ICLR 2021).
- When logged positions come from a good ranker, position is confounded with relevance. A Google study at KDD 2023 ([arXiv 2212.13937](https://arxiv.org/abs/2212.13937)) shows a two-tower position model at 0.3333 NDCG@5 against 0.4191 for the naive model on Web30k under such logging. Hager et al. (SIGIR 2024, [arXiv 2404.02543](https://arxiv.org/abs/2404.02543)) find that unbiased learning-to-rank methods "struggle to consistently improve ranking".
- The industrial gains above come at 10^9 or more training samples, orders of magnitude more than this data holds ([capacity](#model-families-and-capacity)).

## What transferred

These outside ideas are in the final model or in the method. The last column links the experiment folder and, where there is one, the page with the full account.

| Outside idea and source | What ran here | Result | Where |
|---|---|---|---|
| Joint attention over history and candidates ([Avito's own ranker](#avitos-own-ranker)) | attention from each candidate to the user's history, and self-attention across the candidates | History attention +0.0055 on validation (0.37361 against 0.36808, one seed). Candidate attention was neutral in round 1 (0.37386 without, 0.37361 with, both with a position term), and its removal from the final click-label recipe cost 0.0020 / 0.0014 (2-epoch runs, 13 Nov). On day 1 the neural ensemble trained on train alone reached 0.3806 against 0.3700 for the best GBDT. | [neural-ranker-v1](../experiments/neural-ranker-v1/) |
| Retrain with the newest labelled period (RecSys 2024 winner: +0.0097 to +0.0123 test AUC per model) | fine-tune on the two validation days, 5 folds by user | On the first-round model +0.001 (0.37218 to 0.37322), +0.003 with inputs that exist only for validation and test, +0.005 with averaging over epochs (0.37704). The base 0.37218 is a 16-bit read lowered by tied scores, and against the 32-bit read of 0.37385 the averaged form gained +0.0032. In the final recipe the validation gain is about zero, and on public the half-and-half mix of the two branches stays about 0.0005 above either branch alone. | [03-solution.md](03-solution.md#fine-tune-on-the-validation-days); [fine-tune](../experiments/fine-tune/) |
| Random-order flag (Expedia 2013: `random_bool`, which the organisers warned was sometimes wrong) | a detector of random-order feeds inside train, which has no flag | About 13% of train feeds with 50 or more candidates (about 206,000, of which 89,794 above probability 0.9). They became the 30,000-feed holdout and the pool replayed in the cooldown. A 2% share of cooldown batches against 10%: +0.00117 ± 0.00039 / +0.00134 ± 0.00022 (7 paired runs). | [02](02-what-the-data-told-us.md#random-order-feeds-inside-train); [exploration-detection](../experiments/exploration-detection/) |
| Co-visitation counts (OTTO winner: removing them cost 0.0052 Recall@20, as much as removing the neural scores) | item-to-item co-visitation (how often two items occur in the same history) with day cut-offs, as three aggregates, plus leave-user-out popularity counts, as training inputs | +0.0032 / +0.0092 in one paired run. Zeroing one co-visitation input of the final recipe cost 0.0055 / 0.0037. Typed, directional and damped variants each read +0.00024 or less as residuals on the best click-label pool. | [feature-families](../experiments/feature-families/) |
| Leak warning (H&M fifth place: item vectors trained on all data, 0.0441 in cross-validation against 0.0350 on the board) | popularity counts without the same user's later snapshots, and every train row built the way a test row is built | Trees on the uncorrected counts gained +0.025 to +0.052 on the 117 validation feeds whose user has a later feed. | [02](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak) |
| Production order as a training-only signal (PAL, position-aware learning, Guo et al., RecSys 2019; Yandex's context tower, arXiv 2310.03481) | an additive position term dropped at inference, then a second head trained listwise toward the production display order and also dropped at inference | Position term neutral (0.37361 with, 0.37385 without, first-round model). Removing the second head cost 0.0022 / 0.0023 in 2-epoch runs and 0.00135 ± 0.00058 / 0.00128 ± 0.00033 on distilled students (8 seeds). | [neural-ranker-v1](../experiments/neural-ranker-v1/) |
| Retrieval values from TabR, a tabular model that retrieves the nearest training rows and adds a value built from their differences (proposed in the note on tabular models) | for each candidate, the 16 most similar items of the user's own history | Alone +0.00186 / +0.00326 (2 seeds). With the attention gate +0.00278 / +0.00382 (3 seeds). Removal from the final recipe cost 0.0034 / 0.0029. The note advised against TabR's own form, retrieval over training rows, and it was not run. | [optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/) |
| Muon optimiser and a warmup-stable-decay schedule whose cooldown replays in-distribution data (the note on training, from recent language-model practice) | Muon for hidden weight matrices and AdamW for the rest, in place of AdamW with cosine decay and a second training stage | +0.0031 / +0.0033 (3 seeds, 2-epoch runs, validation z 4.6). The replay share had to be retuned when training went from 2 to 10 epochs ([failure 3](07-what-failed.md#3-a-cooldown-dose-tuned-at-2-epochs-and-used-at-10)). | [optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/) |
| Sigmoid gate on the attention output, computed from the query token (recent Qwen models, HSTU) | on all attention modules | +0.00065 on validation, the mean of 4 paired runs (SE about 0.0003). Alone +0.00042 (z 0.9), after the fine-tune +0.00046 (z 0.9). Kept because it is cheap and never read negative. | [optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/) |
| Moving average of weights (Morales-Brotons et al., arXiv 2411.18704) | in the fine-tune, cosine decay with an average of rate 0.99 over 4 epochs, in place of the mean of per-epoch predictions | +0.00112 ± 0.00038 out of fold for members trained with Muon, +0.00025 for AdamW members. Averaging plateau checkpoints of the training stage did the opposite at pool level ([failure 5](07-what-failed.md)). | [optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/) |
| Pretrained sequence encoder (ARGUS, PinnerFormer) | a causal transformer, the trajectory encoder, trained on 2.59 million histories (463.6 million events) to predict the next new item a user opens from the feed. Its scores enter the ranker as inputs. | Zero-shot 0.3590 on validation, the best single signal (history similarity 0.349). The first build of scores for train rows leaked ([failure 2](07-what-failed.md#2-encoder-scores-for-train-rows-first-build)). The adopted fourth build gave +0.00191 ± 0.00058 at 3 epochs (2 seeds) and +0.00077 ± 0.00073 in a 10-epoch removal pair. Fine-tuning the encoder on validation labels lowered it from 0.3536 to 0.350-0.352. | [pretraining](../experiments/pretraining/) |
| Distillation (self-distillation in the note on training; soft pseudo-labels of the KDD Cup 2022 task 1 winner) | 11 Nov: one frozen 2-epoch teacher of the student's own architecture. 14 Nov: a target of half clicks and half the softmax of the mean score of 11 earlier members. | 11 Nov: +0.00064 (z 1.6), closed. 14 Nov: per student +0.00257 ± 0.00047 / +0.00200 ± 0.00028 (4 seeds). The first file of students, which also moved to width 192, scored +0.001032 on public over the best click-label file. | [06](06-what-worked.md#7-distillation-from-an-ensemble-of-earlier-members); [distillation](../experiments/distillation/) |
| Mixture of experts, read as two pools with different inputs (literature round of 15 Nov) | a popularity-light pool (members with one exposure count in place of the 8 popularity inputs; full-view members read all inputs) mixed at weight 0.2, later 0.4 on feeds of fewer than 50 candidates | Weight 0.2: +0.00052 on public (paired SE 0.00041). FD16P20 (16 full-view and 4 popularity-light students trained on all train feeds, weight 0.2, pick 1 until 2026-11-18) scored +0.00041 on public over W16, 16 full-view students trained without the holdout. The short-feed weight scored +0.00032 on public over FD16P20 on 17 Nov, about one paired SE, and is in pick 1, FD16P20@m0.4_0.2. | [popularity-views](../experiments/popularity-views/); [mixing-and-segments](../experiments/mixing-and-segments/); [06](06-what-worked.md#17-popularity-light-weight-04-on-short-feeds) |
| Listwise softmax cross-entropy (Bruch et al., ICTIR 2019; DASALC) | the main loss | Binary cross-entropy read -0.001 on the network (0.37272 against 0.37361). | [03-solution.md](03-solution.md#loss) |
| Seed bagging with an unweighted mean (KKBox; Caruana et al., ICML 2004) | member-equal mean of within-feed z-scores, 20 members per final file | Rank fusion read -0.0020 ± 0.00035 and reciprocal rank fusion -0.0007 ± 0.0003 on validation. | [03-solution.md](03-solution.md#ensemble); [hard-label-pools](../experiments/hard-label-pools/) |

### Why these transferred

- Avito's ranker shape transferred because the data come from that surface. Feeds hold 5 to 300 candidates, consistent with CatBoost's top 300 minus items the user had already seen (an inference), and the ranker reads the same kind of history and item embeddings. Avito's specific production choices did not transfer ([model families](#model-families-and-capacity)).
- Validation is the only labelled sample in test format, which is why the fine-tune paid. Its validation gain shrank to about zero as the train-only model improved, while public kept about +0.0005.
- Random-order traffic paid because train holds about 206,000 such feeds without a flag, and test is all random-order.
- Co-visitation and popularity counts paid because the histories of all splits are large (2.59 million sequences) and counts over them carry signal.
- Muon, the schedule, the gate, retrieval and the weight average act on training and assume nothing about the domain.
- Distillation fits a noisy label. A feed has about one click among about 100 candidates, and the hardest 20% of feeds hold 27.5% of the loss against 27.6% when labels are re-drawn from the model's own softmax. Much of the loss is the label draw, which a soft target smooths.

## What did not transfer

Each idea below closed on the evidence shown. The catalogue in [07-what-failed.md](07-what-failed.md) gives the script or record behind each number.

### Signals this data does not carry

| Outside idea and source | What ran here | Result |
|---|---|---|
| Test-period counts (Expedia 2013 fifth place, arXiv 1311.7679, +0.024 NDCG@38 from counts over train and test. RecSys 2024 winner, AUC 0.8864 with test-period impressions and 0.7699 without) | counts of a candidate in other validation or test feeds and histories, with and without time windows | In-split candidate count -0.0018. Windows of ±1 h cover at most 0.6% of candidates. Appearances in other feeds read -0.00004 ± 0.00020 on the final file. |
| Item click and contact rates as target encodings (Avito ML Cup 2024 top-3 solution, Avito ML Cup 2025 winners, VK RecSys 2024 third place) | train click rate, exposure-normalised rate | -0.0014 to -0.0044 in blends. Plain counts of history events +0.0032 to +0.0035. |
| The 2025 winning recipe (EASE-DAN, RDLAE, SASRec, contact ratios, CatBoost YetiRank) | rebuilt on this data as a tree model, arms of 10 to 88 GPU-minutes | +0.00067 ± 0.00045 over the CatBoost baseline, inside the one-seed spread of 0.00086 (a second seed moved validation by -0.00086 and the holdout by -0.00204). As a member 0.3827 against a bar of 0.3933. |
| Learned item-ID similarity (H&M third place: a BPR score moved the board from 0.03363 to 0.03510) | ALS, item2vec and EASE as residuals on the best click-label pool | 0 to -0.0005 |
| Item-ID table with ID dropout (DropoutNet, NeurIPS 2017: dropout 0.3 to 0.5 keeps warm accuracy and helps cold items) | hashed table (4.2 million rows under click labels, 500,000 for students), ID dropout 0.5 | -0.0022 ± 0.0008 under click labels, and -0.0075 ± 0.0021 on feeds with no covered positive. Students -0.0057 / -0.0049. |
| Outside data: calendar, paydays, weather, news, the exchange rate and Avito's own events for 6-29 Sep | interactions of a day-level fact with category, listing age or posting time, as residuals | Every label-free prior within ±0.0003. An oracle built from validation's own labels +0.00062 ± 0.00036 in sample, +0.00002 ± 0.00018 out of fold. |
| Longer histories (SIM, CIKM 2020: +7.1% click-through rate) | events beyond the 256-event cap, recovered from the user's earlier snapshots. First replayed without retraining, then trained in 16 cooldown re-runs with up to 128 older events from the most recent earlier snapshot (17 Nov). | Replay +0.00117 ± 0.00082 on the 2,358 of 3,235 qualifying holdout feeds that gain at least one event older than the stored 256th, undecided. Trained +0.00150 ± 0.00117 / +0.00001 ± 0.00058 on the 1,132 / 2,159 feeds it reaches (randomization p 0.16), and -0.00013 ± 0.00028 on all validation feeds. |
| Negative feedback from skipped items (WeChat DFN, IJCAI 2020: AUC 0.7824 to 0.7898) | embedding-level probe | Skipped items have a within-feed AUC of 0.58, so they mark interest. Nested -0.0006 to +0.0001. |
| "Next counter" leak (SNA Hackathon 2019 winner; next-click deltas of the TalkingData 2018 winner) | forward echo of feature f5, the item's engagement rate, as other users' later feeds show it | +0.00038 ± 0.00024 (z 1.6). Leak class, not used. |

Test-period counts need items that repeat inside the scored period. Here 91% of validation and 86% of test candidate items occur in one feed of their split, so there is nothing to count. In RecSys 2024 news articles live for hours, and listings do not. Counts over the histories of all splits do carry signal and are in the model as the popularity inputs.

The item-ID methods fail on coverage. About 40 million items occur at least twice in the histories and candidates, and 42.8% of public and 45.6% of private candidate rows hold an item that appears in no history. Items with an ID history fall from 74% of train to 40% of private candidates. The 2025 cup, where these methods won, ranked item groups for Recall@40.

Outside data has no room to work. Every candidate in a feed shares the feed's timestamp, so a day-level fact can act only through an interaction with a candidate attribute, and the oracle of that interaction added +0.0006 ([external-solutions](../experiments/external-solutions/)).

Longer histories reach too few feeds to decide. A first probe read -0.0002 to 0, and the same probe was later found to recover between -23% and +36% of an effect of known size, so its closure counts for nothing. The trained form missed its bar of p 0.05, written before the runs ([04-validation.md](04-validation.md)).

### Model families and capacity

| Outside idea and source | What ran here | Result |
|---|---|---|
| Trees beside the network (RecSys 2024 winner's LightGBM stacker on model scores; ARGUS; Avito's equal rank blend) | trees fitted on train, residual stackers fitted on validation, CatBoost as a pool member (two rounds) | Trees alone 0.368-0.370 on validation. Stacker -0.0001 ± 0.0006. Best CatBoost screen of round 2 +0.00147 / +0.00080 (information gain in nats) against a bar of 0.0016. A tree fitted to the pool's own score stops at 0.3865. |
| Avito's production choices: the ListMLE loss, history filtered to target actions, one stream for history and candidates, a parallel cross branch | each as a change to the recipe | ListMLE -0.00516 (z -6.5, small proxy model). Target-action history -0.00473 / -0.00493, feed-surface history -0.00311 / -0.00266. Single stream -0.00073 ± 0.00051 / -0.00027. Cross branch (DCNv2, GDCN) as a new member family +0.0000 and -0.0004 on validation, +0.0002 and +0.0001 on the holdout. |
| Language-model backbones | Qwen3 0.6B (4 epochs), Qwen3.5 4B and 9B (2 epochs), with LoRA (low-rank adapters), the ranker's inputs as soft tokens, on 150,000 train feeds | -0.0038 / -0.0048 / -0.0053 on validation against the ranker on the same feeds. A randomly initialised trunk ties the pretrained one (-0.00007 ± 0.00060). |
| Capacity blocks: TabM, piecewise-linear numeric embeddings, a mixture-of-experts layer, QK-norm with scaled softmax, forgetting attention, semantic IDs, a propensity head | 2-epoch paired runs, 2 or 3 seeds each | All within ±0.0005 on validation. |
| Mixture of experts in the current form (DeepSeek-V4, Kimi K3: many small routed experts beside a shared expert, bias balancing; SwitchHead: routed value and output projections) and depth 7 at equal size | distilled students of the final recipe, 4 seeds, then 8 seeds for two arms (19 Nov) | 8-pools at the last checkpoint against 8 references: routed feed-forward experts +0.00006 ± 0.00047, depth 7 -0.00001 ± 0.00048 on validation. Routed attention projections, 4 seeds only, against the same seeds of the reference: -0.00144 ± 0.00057 at the checkpoints the holdout selected, -0.00093 ± 0.00056 at the last checkpoint ([optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/#mixture-of-experts-re-tested-on-distilled-students-19-november)). |

Trees are limited by their inputs. A tree fitted to the pool's own score from the 177 tree features stops at 0.3865 on validation, correlation 0.909 with the pool, so no loss or optimiser closes the gap. The network beat the trees on day 1, and the stacker added nothing on the next ranker ([gbdt-and-stacking](../experiments/gbdt-and-stacking/), [catboost-round-2](../experiments/catboost-round-2/)).

In ARGUS the transformer score is one input of a GBDT ranker with about 1,000 features, and the GBDT is the stronger model. Here it is the reverse.

Capacity is limited by data. The note on industrial rankers counted about 5.45 million positives for a model of 0.86 million parameters, where the cited papers report gains at 10^9 or more samples. Width 256, a third block and 8 heads were neutral under click labels, and removing 20% of the users cost a member 0.0028 / 0.0022.

The language-model curve is flat to slightly down over a 16-fold range of trunk size, and random initialisation ties pretraining. The runs used 150,000 feeds, and scaling them to the full train was not tested ([llm-rankers](../experiments/llm-rankers/)).

For ListMLE the records give no tested reason. For the filtered history one reading, not tested, is that the target here is an engagement that starts from the feed: 83 to 87% of positives come back in a later history as a feed-sourced view. A history filtered to target actions drops the views, the event type the label itself turns into.

### Position bias and click models

| Outside idea and source | What ran here | Result |
|---|---|---|
| Examination offset on randomly logged rows (Google, KDD 2023: under random logging their position model read 0.4103 NDCG@5 against 0.3860 for the naive model) | position offset on random-order train rows | +0.00002 / -0.00008 |
| Inverse-propensity weighting (unbiased LambdaMART) | XGBoost | -0.0022 (z -2.2) |
| Cascade click model: a user reads down the list and stops after the last click | on production train feeds, negatives more than 5 positions below the last positive left out of the loss, 16 cooldown re-runs | +0.00004 ± 0.00027 / -0.00000 ± 0.00015 |

The logged positions come from a good ranker, so position is confounded with relevance, the case the Google study describes. The form that worked uses the production order as a training target, through the second head in the table of [what transferred](#what-transferred).

### Training and distillation tricks

| Outside idea and source | What ran here | Result |
|---|---|---|
| R-Drop; sharpness-aware minimisation (SAM) | in the cooldown, weight 1.0 on the R-Drop divergence term and rho 0.05 for SAM | -0.0001 / +0.0006 and 0.0000 / +0.0004 |
| Refit on all of validation for the test prediction (RealMLP paper) | one fine-tune on every validation feed, simulated nested | -0.00025 ± 0.00018 against the mean of 5 fold models |
| Random history dropout (RecSys 2024 winner) | rate 0.3 on distilled students, 4 seeds | -0.00098 ± 0.00058 / -0.00066 ± 0.00033 |
| Pseudo-labels on test feeds (RecSys 2024 winner; gain not reported) | soft targets on 12,000 test feeds per fine-tune epoch at weight 0.3, from fold models that never saw the validation fold, one run on day 1 | -0.0004 out of fold (0.38535 against 0.38580) |
| Function-matching distillation (Beyer et al., arXiv 2106.05237) | random sub-lists of candidates in every batch, teacher scores renormalised, 2 seeds, two rates | -0.00035 ± 0.00066 / -0.00031 ± 0.00038 and -0.00014 ± 0.00068 / -0.00047 ± 0.00039. The sub-lists kept 1.88 positives against 1.22 on real short feeds, so this form was flawed, and the clean form was never run. |
| Teacher temperature (Born Again Neural Rankers, arXiv 2109.15285) | 0.7 and 1.5, 2 seeds | -0.00014 / -0.00032 and +0.00001 / -0.00035 |
| A second generation of students (born-again networks, arXiv 1805.04770: gains saturate after a few generations) | students taught by the 8 students of W8, the first 8-student file, 4 seeds | pool 0.40080 against 0.40134 on validation for the same seeds of the reference; -0.00013 ± 0.00046 against the mean of the 70 possible 4-seed reference pools |
| Teacher labels on unlabelled feeds (the noisy-student pattern in a review of WSDM Cup 2025) | one user fold kept with teacher-only targets, 4 seeds | +0.00069 ± 0.00057 / -0.00017 ± 0.00032, inconclusive |
| Privileged-feature distillation (Yang et al., NeurIPS 2022, arXiv 2209.08754) | a teacher input of leave-one-out item click rates over the train window; separately, a teacher that sees later snapshots | Item rates as a nested residual on the teacher -0.00005 ± 0.00019, against a gate of +0.003. The later-snapshot teacher gained +0.037 ± 0.0025 on covered feeds, all of it after the click, so it copies the label. Not built. |
| Hard-example mining and worst-group losses (online hard example mining, group distributionally robust optimisation; read on 17 Nov) | a check of where the loss sits; both methods stopped at probes. One composite loss ran in 16 cooldown re-runs, each feed weighted by the teacher-student Kullback-Leibler (KL) divergence and surprising clicks down-weighted. | The hardest 20% of feeds hold 27.5% of the loss, and 27.6% under labels re-drawn from the model's own softmax. Composite loss -0.00058 ± 0.00047 / -0.00074 ± 0.00028. |

Every distillation follow-up failed in the same place. Students of one teacher correlate at about 0.97 with each other, so a change that makes one student better usually moves it closer to the others, and the pool does not gain.

The cross-fitted teacher (each train feed scored by three models that never saw its user) gave a pool of 0.40079 / 0.32347 against 0.40134 / 0.32408 for the same seeds of the reference. Against the mean of the 70 possible 4-seed reference pools it is level on validation (-0.00015 ± 0.00047) and lower on the holdout (-0.00059 ± 0.00026); the teacher's softmax is nearly flat, so cross-fitting had little label copy to remove ([distillation](../experiments/distillation/)).

Losses aimed at hard feeds chase the label draw. The loss sits where the click happened to fall, as the 27.5% against 27.6% shows, and the composite loss that down-weighted surprising clicks lost 0.00058 / 0.00074.

### Combining models

| Outside idea and source | What ran here | Result |
|---|---|---|
| Stackers and gates fitted on validation (NVIDIA's third place in RecSys 2022 used three stackers on the validation month. In MIND, Layer 6's tree second stage, second place, took validation AUC from 0.6856 to 0.7733 and test AUC from 0.6826 to 0.6486) | a neural gate that re-weights the 20 members per feed, submitted; 78 second-level combiners | Gate +0.00056 on validation, -0.00035 on public. None of the 78 combiners passed both validation and holdout, and family weights fitted on validation read z -2.85 on the holdout. |

Choosing among near copies on 14,423 validation feeds overfits. The two files chosen that way landed 0.0009 to 0.0010 below their public forecast ([04-validation.md](04-validation.md#noise)).

## How good the priors were

The scan of 2026-11-13 sorted its 211 rows:

- 99 adopted or properly closed, 82 weakly closed, 23 that need data the team did not have and 7 new.
- No technique had a probability above 0.035 of adding +0.0005 at pool level, and the expected gain of the whole re-open queue was +0.0003.
- Distillation, the one technique that paid afterwards, sat in that queue at 0.02. The sweep of 2026-11-12 had listed it among the ideas with no precedent that would transfer here.

Of the 82 weak closures, 26 had tested a different mechanism from the one in the source, 19 had been rejected by argument without a run, 16 by a tree proxy, 9 with the old optimiser, 6 by an MLP (multi-layer perceptron) proxy, 4 at a 2-epoch screen and 2 on one seed. A closure covers only the form that was tested ([09-lessons.md](09-lessons.md), lesson 15).

The round of 2026-11-15 over 82 sources found no untested change to the model with an expected gain above +0.0001 for a file.

## Never tested

Section 6 of [12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) gives a judgement of 19 November on five ideas of this kind, two of them in the first two items below. No experiment stands behind it.

- A graph neural network. An audit closed it: a graph over interactions has no node for the 42.8% / 45.6% of public / private candidate rows whose item appears in no history, and two hops already reach 97% of the rest.
- The trajectory encoder as the starting point of the ranker's history encoder. It failed a gate on stored predictions (a residual on the pool at or below zero) and was closed by argument: the encoder's corpus holds the same events as the labelled train histories, and the published transfers in that setting were about zero.
- A merge with another team, the only way here to ensemble a model of another family. The write-ups report +0.3% to +0.4% relative from ensembles across families, and it was the only outside item the last round estimated above +0.0003 (expected +0.0002 to +0.0004 before knowing the partner). A protocol and a checking script (`lab/src/merge_check2.py`) were written. No final file contains such a model.
- Tree-leaf embeddings as a ranker input, a non-linear token MLP, event-field dropout, a non-additive position embedding and cross-feed negatives, with priors of 0.01 to 0.02 each.
- Distillation on the 43,018 unlabelled validation and test feeds (the noisy-student pattern in a review of WSDM Cup 2025), the one model-side option left open on 18 November ([options](#options-left-open-on-18-november)).

## What the top of this board may use

The public leader has stood at 0.3982556 since 12 November. On the board of 2026-11-18 03:33 UTC it was 0.00305 above pick 1, FD16P20@m0.4_0.2, and places 2 and 3 were 0.00207 and 0.00127 above pick 1. Pick 1 was the best of 14 near copies and sat 0.00041 above their mean, so the leader was about 0.0035 above the team's true level.

That lead was 1.7 to 2.5 cross-team SE, the paired standard error between files of different teams (0.0012 to 0.0018). Files of different teams are less correlated feed by feed than the team's own near copies, so pairing removes less of the feed noise, and their SE is several times the paired SE of 0.0002 to 0.0006 between near copies ([04](04-validation.md#noise)).

On 19 November at 03:34 UTC places 1 to 3 read 0.3982556, 0.3980582 and 0.3968832, and the team's best, pick 2 at 0.3952606, was 9th of 541 teams. Seven teams other than the leader scored above both picks. Pick 2 is the best of 17 near copies, 0.00040 above their mean, and pick 1 is third ([11, Risks](11-final-picks.md#risks)).

On 14 November the records put a better model at P 0.55 and a missed channel at P 0.07, and the expected gains of every outside lever then open added up to about +0.0008 on public ([external-solutions](../experiments/external-solutions/)).

On 18 November three CPU studies took the question up again: a re-read of the past audits against the new board, probes of the data channels on top of pick 1, and a review of what winners elsewhere exploit. A critic then re-checked their numbers, and the probabilities below are the corrected judgement ([`answer_corrected.txt`](../lab/nn_x/w12/leader/critic/answer_corrected.txt)).

### Hypotheses

| Hypothesis | P | Evidence for | Evidence against | Test |
|---|---|---|---|---|
| H-A. Better modelling, or better use of the training data | 0.64 | The top 8 form a continuum, with gaps of 0.00008 to 0.00098. Between 13 and 18 November four teams passed 0.3955, and six other than the leader then scored above pick 1. Distillation added 0.0010 on public here in one day. The model is limited by data: a 3-member pool trained on 80% of the users loses 0.0017 on validation. | Most of the 312 catalogued ideas did nothing for the pool. The error is calibrated: a perfect correction of every bias found in 659 slices would add +0.00035. | Distillation on the unlabelled validation and test feeds, never run ([options](#options-left-open-on-18-november)) |
| H-B. Public luck and best-of-K selection | 0.12 | The leader is 0.5 to 0.8 cross-team SE above place 2. Among 30 to 100 near copies the best file sits above its family mean by +0.0004 (at the per-file sd of 0.00021 measured here) to +0.0013 (at sd 0.0005). | The leading score came from that team's first entry and carries about no selection gain, while pick 1 carries +0.00041. | None needed. Net of selection on both sides, place 2 is +0.0012 to +0.0020 above the team's level, place 3 +0.0004 to +0.0013, and places 4 to 7 are within one cross-team SE. The private order is open. |
| H-C. An entry linked to the organiser or Avito, such as a reference run on production inputs | 0.08 | The score has not changed since 12 November across 3 entries. Avito's production ranker uses about 200 CatBoost features and a transformer trained on up to 10 million feeds. | Six teams working on the anonymised data are only 0.0010 to 0.0029 below it. The board marks no entry as the organiser's. | None from the files |
| H-D. Heavier use of the validation labels | 0.06 | Trees trained on train and validation rows read +0.0015 ± 0.0007 per seed, but as a GBDT at the 0.371 level. The fine-tune on validation gives +0.0005 on public here. | For the pool the gain is about 0. Validation labels replayed in the cooldown read +0.00014, and a refit on all of validation -0.00025. | Nothing new. Expected value at most +0.0003. |
| H-E. A small channel inside the files, shared by many teams | 0.04 | The forward echo of f5 carries information (+0.00457 nats per click). | Apart from the known leak, every channel reads at most +0.00003 NDCG nested on pick 1 and at most +0.00048 on older pools, and none passed. The forward forms use later data, which the team's leak rule forbids. | The probes [below](#probe-results-on-the-best-file) |
| H-F. Fresh item popularity from outside the files, such as crawled view counters | 0.03 | The only input gap of the right size. The 8 popularity inputs are worth 0.0074 NDCG at 1.75 days old, 0.0023 at 5.69 and 0.0009 at 11.23, and public rows are 9.34 days old. Category plus posting minute singles out 47% of test items within the data. | About 2.4 million pages sit behind anti-bot protection. Whether the key is unique on the live site is not measured. The counters are cumulative to the crawl date, so they measure popularity after the feed. | None. Not pursued. |

These probabilities are the judgement of 18 November and were not re-estimated. On 19 November place 2 rose to 0.3980582, 0.0002 below the leader, about 0.1 cross-team SE. That removes the isolation of the leader that H-B and H-C leaned on.

For H-A, section 6 of [12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) names five ideas that never had a fair training run, with a judgement of 19 November on each.

The rule status of H-F is grey at best. Rule 10.3 binds participants to Avito's confidentiality terms, and under rule 8.6.3 the organiser reviews the solutions of the top 10. A view counter also includes the user's own later view of the labelled item, so it breaks the team's leak rule. The team did not pursue it, and the board cannot show whether any entry uses outside data.

Four more explanations were dropped because measurements refute them (P 0.03 together): a merge into the leading row (it appeared with one entry, the rows at places 2 and 3 stayed on the board and the team count rose), probing the public board (the leading score came from that team's first entry), calendar priors (+0.00002 ± 0.00018) and format tricks (the evaluator puts negatives first among tied scores).

### Probe results on the best file

In likelihood the files hold information that pick 1 lacks. In NDCG they do not.

All reads are on top of pick 1 (validation 0.40293). Each fits one conditional-logit weight on half of the users and reads it on the other half, both ways, averaged over 5 random halvings, with feeds of user 0 as single groups. Information is the gain in log-likelihood of the clicks, in nats per click: pick 1 sits at -3.961, a uniform order at -4.308.

| Channel | Class under the team's leak rule | Nats per click | NDCG on validation, nested |
|---|---|---|---|
| Same user's later snapshots: the candidate appears in a later history of the same user (184 rows, 93.5% positive) | leak | +0.0248 ± 0.0030 | +0.00452 ± 0.00047; 0 on public when submitted on 12 November |
| Forward echo of f5: other users' later value of the candidate's f5 | leak class | +0.00457 ± 0.00070 | -0.00015 ± 0.00038 |
| Lost history: events that dropped out of the user's current snapshot (635 validation rows) | legal | +0.00115 ± 0.00040 | +0.00000 |
| As-of channels (other users' activity up to the feed, the backward change of f5, other users' validation clicks on near-duplicates of the candidate as of the feed) and other users' later activity | legal as of the feed; the later forms use later data | at most +0.0003 | -0.00011 to -0.00038, against placebos of -0.00004 to -0.00024 |
| Order and id channels (stored candidate order, embedding index, item id, item-id rank minus f4 rank) | legal | at most +0.00001 | -0.00004 to +0.00003 |

Moving NDCG by +0.003 would take 0.009 to 0.017 nats per click of information at the top of the lists. Lost history holds 7 to 13% of that and the f5 echo 28 to 51%, but the echo sits at the wrong end of the lists. Its content is that an item whose f5 stayed flat was not clicked.

Pick 1 already ranks the lost-history rows high. Their mean relative rank is 0.22, where 0 is the top and all rows average 0.50, and the 24 positives among the 635 rows sit at 0.17. Putting all of these rows first costs -0.00059 ± 0.00022.

A pure-noise column reads -0.00014 ± 0.00005, and flagging 2% of the positives reads +0.0118 ± 0.0007. The harness loses about 0.0001 to 0.0004 per arm, so effects below about +0.0002 cannot be seen.

Channels built from later data cover public rows about as well as validation rows (f5 echo 25.4% against 26.6%), so they do not explain a gap specific to public either. Linking the 43,018 validation and test feeds by history overlap found no hidden identity: 0 links from a guest feed (user 0) to a logged-in user, and only 2 hidden links that start in public.

### Options left open on 18 November

The records of 18 November close the search for channels inside the files. Submissions close on 2026-12-08 at 23:59 MSK (20:59 UTC), and two options keep a positive expected value.

By 19 November neither option had been run or settled. The model-side work of that day tested mixture of experts and depth for the students instead and closed both ([05, phase 9](05-timeline.md#phase-9-mixture-of-experts-and-depth-19-november-from-0400)).

Distillation on the 43,018 unlabelled validation and test feeds has never been run. It would be cross-fitted on user halves, so that no user's later feed trains the model that scores that user, and a final file would have to stay cross-fitted on test users. Its expected value is about +0.0001, with P(gain of +0.001 or more) about 0.02.

The precedent gave teacher targets to a held-out fold of 624,752 train feeds. The pool read +0.00069 ± 0.00057 / -0.00017 ± 0.00032, and a feed's own inputs added nothing (+0.00013 ± 0.00076 on long feeds, -0.00022 ± 0.00106 on short ones). One cross-fitted half here is 3.4% of that fold, which gives +0.00002 if gains scale linearly and +0.0003 if each feed is worth 13 times as much.

The run needs 16 cooldown re-runs at about 0.25 GPU-hours each, the fine-tunes, and a loader for validation and test feeds that the trainer does not have yet. The dose (2 to 5% of cooldown batches) and the 8-seed rule would be declared before the run. That rule passes only 56% of true +0.0005 effects, so a pass is not expected.

A merge with another team has the largest expected gain if a partner agrees: +0.0002 to +0.0004 over the better pool for an unknown partner, up to +0.0008 for one of equal strength, and 0 to +0.0004 for a partner 0.001 stronger. A protocol and a checking script (`lab/src/merge_check2.py`) exist, and the decision rests with the team member who uploads the files.

A merge means passing the solution to the partner, which rule 5.5 allows only until 2026-12-02 12:00 MSK ([11-final-picks.md](11-final-picks.md#what-would-change-the-choice)).

The records also ask for a daily board snapshot with the entry count per row, to catch merges and any new entry from the leader. Code and read-outs are in [`lab/nn_x/w12/leader/`](../lab/nn_x/w12/leader/). There `data_channels/` holds the probes and their numbers (`out/*.json`), `past_audits/board_update.py` the board arithmetic, `outside_practice/p01_posting_key.py` the posting-time key, and `critic/answer_corrected.txt` the corrected judgement. The raw logs are not published.

## Sources and how they were read

The reading ran in eight rounds. None of them found a public write-up of this competition. How the lead session ran the work, and which published harnesses automate such research, is told in [13-how-the-work-was-run.md](13-how-the-work-was-run.md).

- 11 Nov, first hours: write-ups of the closest competitions (RecSys Challenge 2024, Expedia 2013, OTTO, H&M, KDD Cup 2023, the Avito and VK cups) and papers on position bias, neural rankers, sequence pretraining and gradient-boosted ranking.
- 11 Nov, later: four notes on recent tricks in optimisation, tabular models, attention and industrial rankers.
- 12 Nov: a sweep of recent competitions, industrial ranking papers and Kaggle training tricks, which gave 44 ideas, each matched against what had run.
- 13 Nov: a scan of 211 technique rows from about 120 public sources, each checked against the runs.
- 14 and 15 Nov: the distillation literature after the distillation result, then mixture of experts and graph networks, then 82 more web sources.
- 17 Nov: losses for hard examples and click models, before the last round of loss changes. Only a summary of these notes was kept, in the decision ledger, which is not published.
- 18 Nov: a review of what winners elsewhere exploit, one of the three studies of the gap to the top of the board ([above](#what-the-top-of-this-board-may-use)).
- 19 Nov: the mixture-of-experts designs of DeepSeek-V4 and Kimi K3 and the SwitchHead attention, read for the re-test of mixture of experts on the students.

The recap table was checked against the web on 2026-11-18. That check added the names of the first two RecSys 2025 teams, taken from the fourth-place paper.

Some sources were read only in part. The ACM full texts of the RecSys Challenge 2024 papers returned an error, so slides, code and abstracts were used. PAL was read as a search result, the function-matching and privileged-features papers as abstracts, and OneRec through a secondary summary. Kaggle write-ups were read through the HF mirror, and the H&M fifth-place figures (0.0441 against 0.0350) could not be re-checked on 2026-11-18.

A source is cited the way the notes read it. The notes are `deep_*.md`, `research_*.md`, `nnx_research_*.md`, `RESEARCH_SWEEP_1112.md`, `EXTERNAL_SCAN_1113.md`, `TOP1_GAP_EXTERNAL_1114.md`, `LIT_1115/` and `TRICKS_1115/` under [`lab/reports/`](../lab/reports/). The studies that measured outside solutions and outside data on this data are in [experiments/external-solutions](../experiments/external-solutions/). The ideas that changed the model have their own folders, linked in the tables above.
