# The final solution

In brief:

- Each final file averages small transformer rankers (2.1 million parameters at width 192) that score every candidate of a feed against the user's history.
- Every ranker in these files is a student: trained toward half clicks and half the averaged scores of 11 or 25 earlier rankers, then fine-tuned on the two validation days.
- Pick 1, FD16P20@m0.4_0.2, scored 0.3952092 NDCG on the public board and 0.40293 on validation. Pick 2, FDX16P20:ps, scored 0.39526061 (rank 9 of 541 teams on 2026-11-19 03:34 UTC) and 0.40286.

The earlier rankers that provide the averaged scores are the teacher. They were trained on clicks alone, and this page calls such rankers click-label rankers. Two kinds of student are pooled. A full-view student reads all 34 pre-computed inputs. A popularity-light student reads 27: one clipped exposure count replaces the 8 popularity inputs (3 co-visitation scores and 5 leave-user-out counts; [Inputs](#inputs) explains each).

The first final pick, FD16P20@m0.4_0.2, pools 16 full-view and 4 popularity-light students, all of width 192 and all trained on the 3,151,208 train feeds. The popularity-light pool has weight 0.4 on feeds of fewer than 50 candidates and 0.2 on the others. It scored 0.3952092 on public, the team's best score until 2026-11-19, and 0.40293 on validation.

The same 20 members at weight 0.2 on every feed make FD16P20 (public 0.394892, validation 0.40271), the first pick until 2026-11-18.

The second pick, FDX16P20:ps (public 0.39526061, validation 0.40286), keeps the 4 popularity-light students at weight 0.2 and replaces the 16 full-view students by 16 that also train on pseudo-feeds built from user histories. It replaced W8P20 (public 0.3950094, validation 0.40214) on 2026-11-19. [The final files](#the-final-files) compares the four.

How to read the numbers:

- Scores, splits and notation follow [04-validation.md](04-validation.md); terms are in [glossary.md](glossary.md).
- NDCG (normalised discounted cumulative gain) is the competition metric: full-list NDCG per feed, averaged over feeds. Validation is the 14,423 labelled feeds of Sep 22 and 24. The holdout is 30,000 random-order train feeds that held-out members never train on. Public is 8,562 test feeds.
- A result written a / b is validation / holdout NDCG, and the members compared on the holdout did not train on its feeds. "±" is one paired standard error over feeds.
- Lab code names appear once, in brackets, where a file or directory carries them. Commands are in [10-reproduce.md](10-reproduce.md).

## The pipeline at a glance

![Nine stages from the competition data to the submission file (flatten, exploration probability, candidate inputs, teacher rankers, teacher files, students, fine-tune, ensemble and file, pre-submission gate), each with its hardware and its time in the original run; the GPU stages add up to 151.6 GPU-hours, the 20 students to 81.8](figures/pipeline.svg)

1. Flatten the raw parquet into arrays that load onto a GPU as they are.
2. Estimate for every train feed its exploration probability, the probability that it was shown in random order ([02-what-the-data-told-us.md](02-what-the-data-told-us.md#random-order-feeds-inside-train)).
3. Build 34 inputs per candidate for all three splits, and 42 more for validation and test only ([Inputs](#inputs)).
4. Train 25 click-label rankers, the teacher members.
5. Score every train candidate with them and average the scores into two teacher files ([Distillation](#distillation)).
6. Train the students for 10 epochs toward a target that is half clicks, half teacher. The full-view students of the second pick also train on pseudo-feeds ([The pseudo-feed students](#the-pseudo-feed-students-of-the-second-pick)).
7. Fine-tune every student on the validation days in 5 folds by user ([Fine-tune on the validation days](#fine-tune-on-the-validation-days)).
8. Pool the members, mix the two input views and the two branches (fine-tuned and train-only predictions), and write within-feed ranks ([Ensemble](#ensemble)). The popularity-light weight is 0.2, or in the first pick 0.4 on feeds of fewer than 50 candidates; the branches mix 0.5 / 0.5.
9. Check the file with the pre-submission gate ([04-validation.md](04-validation.md#the-pre-submission-gate)).

## Inputs

### What the ranker reads

Per feed the ranker reads up to 300 candidates and the 256 newest history events. A candidate carries its 128-dimensional int8 content embedding, its category and the five organiser features f1 to f5. A history event carries the item's embedding and category, the event type and the event source. The user is one 128-dimensional int8 embedding.

The ranker gets no display position (validation and test have none), no timestamp and no item identifier. A hashed item-ID embedding table was tried twice. It cost 0.0022 ± 0.0008 on validation under click labels and 0.0057 / 0.0049 for a distilled student (2 seeds).

### Columns computed inside the model

The model computes 63 columns from the raw fields on the GPU, inside each batch (`avito_ranker/model/dense.py`; lab original `dense_features` in `lab/nn_x/w5/model.py`). M is the number of candidates in the feed and H the number of history events.

| Columns | Count |
|---|---|
| f1 to f5: raw, within-feed z-score and within-feed percentile | 15 |
| Embedding-status columns | 2 |
| Cosine and dot product with the user embedding | 2 |
| Summaries of the cosine to the history items | 6 |
| Largest cosine to the history per event type | 10 |
| Category affinity with the history | 13 |
| Feed context: the category's share of the feed, log M, log(1 + H) | 3 |
| Within-feed z-score and percentile of five of these columns | 10 |
| Flags for the pooled anonymous user (user 0), alone and times log(1 + H) | 2 |

### The 34 pre-computed inputs

The pre-computed inputs are arrays built before training, one float per candidate, for train, validation and test (`avito_ranker/features/`). The first two families are the 8 popularity inputs.

| Family | Inputs | Arrays in the lab tree |
|---|---|---|
| Co-visitation between the candidate and the feed's history | 3 | `seq_cv_invd20`, `seq_cv_pop`, `seq_cv_norm` |
| Leave-user-out history counts of the item, total and for event types 1 to 4 | 5 | `it_hist_all_luo`, `it_hist_et1_luo` to `it_hist_et4_luo` |
| Near-duplicate structure among the feed's candidates | 10 | `ctx_nd_*` |
| Competition of the candidates for the user's recent history items | 10 | `ctx_hn_*` (6), `ctx_hs_*` (4) |
| Scores of a self-supervised trajectory encoder, with a presence flag | 6 | `pre2_fe_z`, `pre2_new_z`, `pre2_fe_s`, `pre2_new_s`, `pre2_hmax32`, `has:pre2_fe` |

### Four rules for every input

The leak incidents told in [04-validation.md](04-validation.md#leak-incidents) led to four rules, and every input had to pass them:

1. Nothing from a later snapshot of the same user.
2. No label of the row's own feed.
3. No linking of user-0 feeds (user 0 pools the visitors who were not logged in).
4. Train rows built exactly like validation and test rows.

The first rule exists because a history is a snapshot and the click shows up in the next one: 92.1% of a feed's positives are in the same user's next history. An overlay built on that scored 0.392819 on public against 0.392820 for its base file ([02-what-the-data-told-us.md](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak)).

### How the families are built

Co-visitation counts how often the candidate occurs within 20 positions of one of the feed's history items in other users' de-duplicated histories, weighted by distance and by how recent the history item is. Sequences of the same logged-in user are skipped. A train row of day d reads train histories from before day d. A validation or test row reads all train histories plus validation and test histories with an earlier timestamp.

The leave-user-out counts are the number of times the item occurs in the history snapshots of all users in all splits, minus the occurrences that come from a later feed of the same user. They still include other users' later snapshots. That leaks no label, but it makes the counts of a train row fresher than those of a test row, which is why the second input view exists ([The two input views](#the-two-input-views)).

The two in-feed families read the feed's own embeddings and nothing else:

- The near-duplicate family describes the similarity graph among the candidates: the largest cosine to another candidate, and the user similarity, history similarity and f1 of a candidate's nearest neighbours.
- The competition families ask, for each of the user's last 32 to 128 history items, which candidate of the feed is nearest, and give each candidate its share of that attention.

### The trajectory encoder

The trajectory encoder ([experiments/pretraining](../experiments/pretraining/)) is a self-supervised causal transformer (width 384, 4 layers) over history sequences, a separate model from the ranker. One head predicts the next new item the user opens from the feed, the other the next new item of any kind. Its score alone ranks validation at 0.3590.

For every logged-in user, the encoder's corpus holds the latest train snapshot and the earliest validation or test snapshot. It therefore never sees a snapshot taken after a validation or test feed of the same user.

Train rows are scored by encoders trained on three time windows (Sep 9-11, 12-16, 17-19). Each sees a user's latest snapshot before the window and the earliest one inside it. The 864,519 train feeds of Sep 6-8 have no score and a presence flag of 0.

Input building went wrong twice here ([07-what-failed.md](07-what-failed.md#2-encoder-scores-for-train-rows-first-build)). The first train-row build leaked the label (holdout +0.006 to +0.0085, validation -0.005 to -0.009) because the encoder's item table held no train candidates. After the fix, a clip at ±10 in the loader left 76% to 92% of the raw scores on the clip value. The rescaled copies (`pre2_fe_s`, `pre2_new_s`) repair that.

### From inputs to the dense block

The loader applies log1p to the 8 popularity inputs, clips z-scores to ±5 and the rest to ±10, and stores float16. The model adds the within-feed percentile and z-score of five of the inputs. It drops the raw value of three encoder columns whose scale differs between splits.

With the 63 computed columns this gives the dense block: 63 + 34 - 3 + 10 = 104 columns, standardised with train statistics and clamped to ±6.

### What each family is worth

Each family was removed from the click-label recipe in paired 2-epoch runs (seeds 23 and 24).

| Family removed | Cost, validation / holdout |
|---|---|
| The 8 popularity inputs | 0.0023 / 0.0074 |
| Near-duplicate structure | 0.0014 / 0.0003 |
| Trajectory-encoder scores | 0.00128 ± 0.00061 / 0.00054 |
| The two competition families | 0.0005 / 0.0000 (inconclusive; kept) |

## The ranker

![The ranker: the user vector and up to 256 history events become a user token and history tokens of width 192, encoded by one self-attention block; each of up to 300 candidates becomes a token built from its embedding and a dense block of 104 columns, adds values retrieved from its 16 most similar history items, and passes twice through cross-attention to the history and self-attention over the feed's candidates; a score head and a training-only auxiliary head follow, with widths](figures/architecture.svg)

The model is `avito_ranker/model/` (lab original: class `Ranker` in `lab/nn_x/w5/model.py`). Width d is 192 with 4 heads, a feed-forward width of 2d, dropout 0.1 and pre-norm blocks. It has five attention blocks and 2,118,848 parameters.

In the sketch below, X is the dense block, e a unit-normalised content embedding and D the domain embedding. E_* are learned embeddings, LN is layer normalisation, MLP a small multilayer perceptron, Z a zero-initialised output layer and s_head a learned scale per attention head.

```text
user token       u    = Linear(128 -> 192)(unit user embedding)
history token    h_j  = LN( Linear(128 -> 192)(e_j) + E_type + E_source + E_category + E_recency[j] )     j = 0 (newest) .. 255
history encoder  [u, h] <- one self-attention block over [u, h_0 .. h_255]
candidate token  c_i  = LN( Linear(256 -> 192)([e_i, e_i * e_user]) + E_category + MLP(104 -> 384 -> 192)(X_i) ) + D
retrieval        c_i += Z( sum_j p_ij * v_ij )            j over the 16 history items with the largest cos(e_i, e_j)
2 ×              c <- cross-attention(c -> [u, h]),  attention logits + s_head * cos(e_i, e_j)
                 c <- self-attention over the candidates of the feed, no positions
score            s_i  = MLP(LN, 192 -> 192, GELU, 192 -> 1)(c_i + D) + Linear(104 -> 1)(X_i)             in float32
auxiliary head   a_i  = MLP of the same shape on c_i + D                                                 training only
```

### Attention to the history

Attention from each candidate to the user's history is the largest single component ([What each part is worth](#what-each-part-is-worth)). The cosine between candidate and history item, times a learned scale per head, is added to the attention logits. Attention therefore starts from the similarity that Avito's own embeddings already carry.

### Retrieval of the 16 most similar history items

Retrieval gives each candidate a direct look at its nearest history events and at what the user did with them. For the 16 history items most similar to the candidate:

```text
k        = Linear(128 -> 64)(e)                                   one projection for candidates and history items
logit_ij = -|k_i - k_j|^2 + a * cos(e_i, e_j) + beta[type_j]
p_ij     = softmax over the 16 items and one learned "sink" slot that contributes nothing
v_ij     = LN( E_type + E_source + E_recency_bucket + MLP(64 -> 128 -> 192)(k_i - k_j) )
```

The output layer Z starts at zero, so the block is a no-op at initialisation. About 90% of its effect comes from the difference term MLP(k_i - k_j).

Sixteen items are enough. Retrieval depth 32 read -0.0002 / -0.0003 in 10-epoch pairs and -0.00098 / -0.00043 for distilled students. A variant with one slot per distinct item gained 0.00117 on validation at 2 epochs and 0.00036 ± 0.00068 at 10, and was dropped.

### Attention over the feed

The second half of each block is self-attention over the feed's candidates without positional encoding. The score of a candidate then depends on what else is in the feed. The model is equivariant to the order of the candidates: scores move by 1e-4 when the feed is re-shuffled. That fits the task, since test feeds are shuffled before display and the metric is computed per feed.

Every attention output is multiplied by a sigmoid gate computed from the query (gated attention). The gate was kept because it is cheap and its measured effect was never negative.

### The domain embedding

The domain embedding tells the model which kind of feed it is scoring. Most train feeds are in production order, and their clicks depend on position (11.8% positive at position 0, 0.92% at position 200). Validation and test are in random order.

The embedding has three rows. In training a feed gets the mix (1 - p, p, 0) of the production row and the random-order row, where p is its exploration probability. For feeds with fewer than 50 candidates p is set to 0, because the detector is unreliable there. Train-only inference on validation and test uses the random-order row. The fine-tune uses the third row, initialised from the second.

The vector is added to every candidate token and again before the score head. It has 576 parameters and was kept although its measured effect is within noise.

### The score head

The score is the head MLP plus a linear function of the 104 dense columns, so f1 to f5 and the pre-computed inputs reach the output without passing through attention. This skip was in the first model and was never removed in a test, so it has no measured value.

Both terms are computed in 32-bit floats, while the rest of the model runs in bfloat16 (a 16-bit float format). At typical logit sizes a 16-bit logit moves in steps of 1/64. That creates ties within a feed, and the metric orders tied candidates with the negatives first.

### What each part is worth

| Part | Measured effect | Conditions |
|---|---|---|
| History tokens | validation 0.36808 without them, 0.37361 with them | first model ([experiments/neural-ranker-v1](../experiments/neural-ranker-v1/)), one seed; the history-similarity columns of the dense block stayed in |
| Cosine bias in the history attention | +0.0024 on validation when it went in; removing it cost 0.0004 / 0.0005 (inconclusive) | one seed; the removal was from the final click-label recipe, by when retrieval had its own cosine term |
| Retrieval alone | +0.00186 / +0.00326 | 2 seeds |
| Retrieval with gated attention | +0.00278 / +0.00382 | 3 seeds |
| Removing retrieval and gated attention | -0.0034 / -0.0029 | final click-label recipe |
| Self-attention over the feed's candidates (set attention) | neutral in the first model set; removing it cost 0.0020 / 0.0014 | final click-label recipe |
| Gated attention | +0.00065 on validation | mean of 4 run pairs |
| Domain embedding | neutral when it went in together with the user-0 columns; removing it cost 0.0004 / 0.0000 (inconclusive) | |
| Score head in 32-bit floats | validation from 0.3705 to 0.3736 | |
| Linear skip from the dense block | never removed in a test | |

### Capacity

More capacity did nothing under click labels. Width 192 came back with distillation ([Distillation](#distillation)).

| Test under click labels | Result, validation / holdout |
|---|---|
| 7 members of width 192 (6 heads) against 7 of width 128, trained the same way | mean member 0.39371 / 0.31843 against 0.39460 / 0.31911 (-0.00089 / -0.00068) |
| The same two 7-member pools | 0.39853 / 0.32267 against 0.39846 / 0.32263, a tie |
| 5 cooldown re-runs at width 192 against 5 at width 128 (2026-11-14) | -0.00087 / -0.00106 |
| Three blocks, eight heads, a second history layer | level or worse |

The test of 2026-11-14 re-ran only the cooldown, the last 20% of training ([Optimiser and schedule](#optimiser-and-schedule)). At the end of the constant phase before it, the wider model's loss was lower on its own training feeds (z -5.5) and not on held-out feeds (z +1.5).

Capacity was tested once more on the distilled students, on 19 November. Three arms ran with 4 seeds each: routed feed-forward experts beside the dense layer (a mixture of experts), routed value and output projections in the cross-attention blocks, and 7 blocks in place of 5 at about equal size. Paired by seed at the last checkpoint they read +0.00009, -0.00093 and -0.00016 ± 0.00056 on validation.

At the checkpoints the holdout selected the screen had read -0.00086, -0.00144 and -0.00069, but all-data students keep the last step. At 8 seeds against the 8 reference students, the experts read +0.00006 ± 0.00047 and depth 7 -0.00001 ± 0.00048 on validation, and both were closed ([experiments/optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/)).

## Loss

For a feed with clicks y and scores s, the main loss is the cross-entropy between softmax(s) and a target distribution over the feed's candidates. A click-label member uses y / sum(y); a student mixes in the teacher ([Distillation](#distillation)). Binary cross-entropy per candidate was 0.001 worse on validation (0.37272 against 0.37361).

```text
target_i = 0.5 * y_i / sum(y) + 0.5 * softmax(t)_i                     t = teacher logits of the feed
L_main   = ( sum over the feeds of the batch of  -sum_i target_i * log softmax(s)_i ) / 512
L_aux    = weighted mean over the feeds of the batch, weight (1 - p) per feed, of
           -sum_i softmax(-pos / 10)_i * log softmax(a)_i
L        = L_main + w_aux * L_aux          w_aux = 0.5, falling to 0.1 with the learning rate in the cooldown
```

Here p is the feed's exploration probability, pos the display position of a candidate in a train feed and a the auxiliary head's output.

### Equal weight per feed

Dividing by 512, the largest number of feeds a batch can hold, gives every feed the same weight. Batches hold feeds of similar length, up to 32,768 candidates. A batch of 300-candidate feeds therefore has about 109 feeds, and a batch of feeds with up to 64 candidates has 512.

A batch mean would weigh the long feed 4.7 times as much as the short one, while the metric is an unweighted mean over feeds. Equal weight per feed added 0.0020 / 0.0006 in one paired run. Going back to the batch mean cost the final click-label recipe 0.0015 / 0.0010.

### The auxiliary loss toward the display order

A second head on the same token learns the order in which Avito's ranker displayed the feed: a listwise cross-entropy toward softmax(-position / 10). Each feed is weighted by 1 - p, so random-order feeds, whose positions carry nothing, drop out. The head is thrown away after training.

| Test | Result, validation / holdout |
|---|---|
| Against a regression on the position | +0.0024 / +0.0027 (one paired run) |
| Removed from the final click-label recipe | -0.0022 / -0.0023 |
| Removed from a pool of 8 distilled students | -0.00135 ± 0.00058 / -0.00128 ± 0.00033 |
| Temperature 30 in place of 10 | -0.0009 / -0.0011 |

The auxiliary head can help a model that is tested on random order only through the layers the two heads share. Its own output is useless as a score. On validation it reaches 0.3036 NDCG, where a random order scores about 0.289 and the main head 0.3977 (12 members pooled). Its within-feed correlation with the main score is 0.03.

The team reads the gain as regularisation of the shared layers by a second task with a dense target. Every candidate of a production feed has a position, while 66.6% of train feeds have exactly one click.

One observation fits that reading. With this head and the domain embedding in place, an explicit correction for the way production feeds are cut off after the last interaction added nothing to the network (-0.0005 ± 0.0005 on validation). It had added 0.0022 to 0.0031 in a tree-model proxy. The reading was not tested further.

## Optimiser and schedule

Two optimisers follow one schedule (`avito_ranker/training/`). Muon, an optimiser that orthogonalises the update of each weight matrix, updates the hidden matrices. AdamW updates the rest.

| | Muon | AdamW |
|---|---|---|
| Updates | the 46 hidden weight matrices: 2,022,400 of the 2,118,848 parameters | embeddings, biases, norms and single-output layers |
| Settings | momentum 0.95, weight decay 0.02 | betas 0.9 / 0.98 |

The key and value projections of every attention block are separate matrices, so Muon orthogonalises each on its own. Muon's step is scaled to AdamW's update size (`match_rms_adamw` in PyTorch), so one learning rate serves both. Gradients are clipped at norm 1.0.

```text
lr(step) = 2e-3 * (step + 1) / 500                                        step < 500
         = 2e-3                                                           500 <= step < 79,872
         = 2e-3 * (0.03 + 0.97 * (1 - sqrt((step - 79,872) / 19,968)))    79,872 <= step < 99,840
```

A run is 10 epochs of 9,984 steps on all data (9,835 steps when the holdout is kept out). The schedule warms up for 500 steps, holds a constant rate, then decays in the cooldown, the last 20% of the steps. Against AdamW with cosine decay and a second stage on random-order feeds, Muon with this schedule gave 0.0031 / 0.0033 over 3 seeds in 2-epoch runs ([experiments/optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/)).

### Ten epochs

Ten epochs pay at pool level only. One member gains 0.0006 / 0.0007 from 10 epochs over 2 under this optimiser. The members of long runs differ more from each other (within-feed correlation between seeds 0.93 to 0.94 against 0.978), and a 6-seed pool gains 0.0040 / 0.0031.

Longer runs lost. Runs of 15, 20 and 30 epochs cost a member 0.0006, 0.0008 and 0.0016 on validation, and a batch 4 times larger cost 0.0026 / 0.0021.

### The cooldown and its random-order batches

The score arrives in the cooldown: a 7-seed pool gains 0.0031 ± 0.0006 on validation between the end of the constant phase and the end of the run.

During the cooldown each step is, with probability 0.02, a batch drawn from the random-order pool. That pool is the 164,118 train feeds with exploration probability above 0.5 and 50 or more candidates (134,118 when the holdout is kept out; [02-what-the-data-told-us.md](02-what-the-data-told-us.md#random-order-feeds-inside-train)). The dose was found on click-label members ([experiments/rounds-5-7](../experiments/rounds-5-7/)).

| Share of cooldown steps from the random-order pool | Result |
|---|---|
| 30%, tuned on 2-epoch runs | all six 10-epoch members collapsed in the cooldown, from about 0.391 to 0.364 on validation ([07-what-failed.md](07-what-failed.md#3-a-cooldown-dose-tuned-at-2-epochs-and-used-at-10)) |
| 10% | in re-runs of the cooldown from a stored checkpoint, the best step fell between 5,000 and 10,000 of 19,670, and the runs then lost 0.0026 / 0.0026 |
| 4% | -0.00077 on validation against 2% |
| 2%, kept | the best step is the last one or close to it; against 10%, a member gains 0.00117 ± 0.00039 / 0.00134 ± 0.00022 (7 runs) |
| 1% | -0.00035 on validation against 2% |
| 0% | ties with 2% on validation and reads higher on the holdout (+0.00002 ± 0.00025 / +0.00057 ± 0.00014); loses 0.0023 ± 0.0005 on the validation feeds whose positives are all cold items; its pool reads -0.00042 on validation |

A cold item is one that was never a train candidate before the feed's day. Because of the loss on cold-item feeds and the lower pool, the share stayed at 2%.

### Checkpoint choice

A member that holds out the 30,000 feeds is evaluated on them every 2,500 steps and keeps the best checkpoint from step 12,000 on, so no validation label is used in the train stage. A member trained on all data has nothing to select on and keeps the last step.

## The two input views

A second kind of member reads a popularity-light view ([experiments/popularity-views](../experiments/popularity-views/)). The 8 popularity inputs are removed and one input is added: an exposure count log1p(min(n, 50)). Here n is the number of times the item was shown as a train candidate: before the row's own day for a train row, on all train days for a validation or test row. That leaves 27 pre-computed inputs and a dense block of 93 columns (`xp_imp_c50` in the lab tree).

### Why a second view

The 8 popularity inputs go stale. They count what users did with an item, almost all of it in train, which ends on Sep 19. A warm item was last a train candidate 1.75 days before a holdout row on average and 11.23 days before a private row, a staleness no member saw in training.

The holdout's inputs were rebuilt at private-like staleness. On the 13,424 feeds where the rebuild passes its check against test rows, the full-view pool lost 0.0033 ± 0.0008 and the popularity-light pool 0.00049 ± 0.00038 ([02-what-the-data-told-us.md](02-what-the-data-told-us.md#cold-items-and-stale-popularity) has the measurements; [experiments/popularity-staleness](../experiments/popularity-staleness/)).

### Weight 0.2 in every file

Where the inputs are fresh, a popularity-light student is the weaker one. At width 128 it scores 0.39821 / 0.31625 against 0.39911 / 0.32229 for a full-view student. So the view enters as a small share of the pool: 0.2 in FD16P20, W8P20 and the second pick.

The weight was fixed on 2026-11-14, before any score of such a file existed. It lies between an earlier estimate of 0.15 and the 0.29 share (8 of 28 members) that popularity-light rankers had in the best click-label file. It was not scanned at the time.

Against the same pool without it, the admixture read +0.00049 ± 0.00031 on validation and +0.000523 on public (paired standard error 0.00041). The holdout read it at -0.00063 ± 0.00022, because its inputs are 1.75 days old ([04-validation.md](04-validation.md#rules-written-before-the-read)).

### Extensions that did not help

| Extension | Validation | Public |
|---|---|---|
| Four popularity-free students, which see neither the 8 popularity inputs nor the exposure count | failed their replication: above 70 of 70 control pools on the first draw, 16 of 70 with fresh seeds ([04-validation.md](04-validation.md#rules-written-before-the-read)) | FD16P20vf (FD16P20 plus four such students) scored 0.394630, 0.00026 below FD16P20 |
| 8 or 12 popularity-light students in place of 4 | -0.00021 ± 0.00017 and -0.00016 ± 0.00018 | not scored |
| Weight 0.4 on feeds of 50 or more candidates | +0.00026 | 0.00015 below FD16P20 |
| A1: on feeds of 50 or more candidates, the popularity-light part of the fine-tuned branch taken from the train-only predictions | +0.00017 ± 0.00014 against the first pick | 0.00017 below the first pick |
| A cooldown of the full-view students on artificially stale inputs | passed at pool level; -0.00047 ± 0.00035 in the file ([07-what-failed.md](07-what-failed.md#19-a-cooldown-on-stale-popularity-inputs)) | not scored |

### The feed-size rule: 0.4 on short feeds

One extension went into the first pick: weight 0.4 on feeds of fewer than 50 candidates and 0.2 on the others. On 2026-11-17 a single weight for every feed of FD16P20 was read on validation (first row of the table). The long-feed side had already met the board (weight 0.4 on feeds of 50 or more candidates, above), so only the short-feed side was built.

Validation NDCG of the 20 members of FD16P20 at each popularity-light weight:

| Weight | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|---|
| Same weight on every feed, read before the public score | 0.40264 | 0.40271 | 0.40287 | 0.40319 | 0.40282 | 0.40261 | |
| Weight on feeds of fewer than 50 candidates, 0.2 on the others, read after it | | 0.40271 | 0.40266 | 0.40293 | 0.40251 | 0.40232 | 0.40184 |

Against FD16P20 the rule read +0.00022 ± 0.00024 on validation, +0.00027 in the public-mix view and +0.00033 in the private-mix view. These views re-weight validation to the feed sizes and cold shares of the public and private parts of test ([04-validation.md](04-validation.md#the-witnesses)). All of the validation gain came from the 5,893 validation feeds of fewer than 50 candidates.

The expected public gain was about +0.0001. The file scored 0.3952092, +0.00032 at a paired standard error of 0.00031. The rule became part of the first pick after that score ([11-final-picks.md](11-final-picks.md#why-pick-1)).

The second row of the table was read after the public score. On it 0.4 is a single high point of this draw of four popularity-light students, with no smooth optimum around it. Eight or twelve popularity-light students under the same rule read 0.00057 and 0.00039 lower than the four.

Each branch of the first pick alone reads below the same branch of FD16P20 (fine-tuned 0.40139 against 0.40190, train-only 0.40208 against 0.40213), so the gain appears only in their mix. On long feeds the weight stayed at 0.2, where 0.1 and 0 read 0.00025 and 0.00068 lower on validation.

### Where the staleness cost sits

In the fine-tune, seven of the 8 removed inputs come back through the extras, so the two views differ in the train stage only. Stale validation is validation re-predicted with the popularity inputs at private-like staleness. It shows that the train stage is where the cost sits.

| Pool, on stale validation | Change |
|---|---|
| Fine-tuned popularity-light pool | +0.00026 ± 0.00025, no loss |
| Fine-tuned full-view pool | -0.00173 ± 0.00053 |
| Train-only full-view pool | -0.00176 ± 0.00052 |

## Distillation

By 2026-11-13 the team's best file was the best click-label file (C3 in the lab records). It held the 20 rankers of the best click-label pool (G3, public 0.392820) and 8 popularity-light click-label rankers, 28 in all, with public 0.393454 and validation 0.40101. Such a pool was about 0.004 better on validation than any of its members.

One click among about 100 candidates is a noisy label, and the pool's averaged score is a smoother one. Distillation trains a single model, the student, on both ([experiments/distillation](../experiments/distillation/)).

Distillation had been tried once before. On 2026-11-11 a single 2-epoch teacher read +0.00064 on validation (z 1.6), and the idea was closed. The ensemble teacher of three days later gave the largest step of the team's public score after the first day. Nothing in the setup caught that closure in time ([13-how-the-work-was-run.md](13-how-the-work-was-run.md#what-the-setup-lacked)).

![How a student's training target is made: 25 rankers trained on clicks score all 246.7 million train candidates, their logits are centred per feed and averaged over 11 or 25 rankers, and the target is 0.5 times the clicks plus 0.5 times the softmax of the teacher; one student gained 0.00257 on validation and 0.00200 on the holdout, the first file of students 0.00103 on public](figures/distillation.svg)

### The teacher

The teacher members are 25 rankers of four kinds. All were trained on clicks without the 30,000 holdout feeds, and all are taken as train-only checkpoints, so no validation label and no fine-tuned model reaches a teacher.

| Teacher members | Count | In the 11-member teacher |
|---|---|---|
| Transformer rankers, width 128, cooldown re-run with 2% random-order batches (`r6_cd_m02_r101-107`) | 7 | yes |
| The same recipe trained in one run (`w5_e10_base_s951-954`) | 4 | yes |
| Rankers with a bidirectional GRU (gated recurrent unit) as history encoder (`v23_gru_pre2s_s501-506`) | 6 | no |
| Popularity-light rankers, width 128 (`abl_e10_pop1_s951-958`) | 8 | no |

Each member scored all 246,745,836 train candidates. The logits were averaged over the 11 or over all 25 members and centred within each feed. That gives one float per train candidate and a file of 0.99 GB per teacher (`avito_ranker/training/teacher.py`).

A student's target for a feed is 0.5 × y / sum(y) + 0.5 × softmax(t), with no temperature ([Loss](#loss)). Everything else in the recipe stays as it was, including the auxiliary loss.

### What the students gained

A student of the 11-member teacher gained 0.00257 ± 0.00047 / 0.00200 ± 0.00028 over the same recipe trained on clicks (width 128, 4 seeds). Levels of train-only predictions:

| Model | Validation / holdout |
|---|---|
| One click-label member | 0.39648 / 0.32022 |
| The 11-member teacher pool | 0.40054 / 0.32371 |
| The 25-member teacher pool | holdout 0.32286 |
| One full-view student, width 192 | 0.39935 / 0.32263 |
| Eight such students pooled | 0.40157 / 0.32456 |

Eight students are above the ensemble that taught half of them. After the fine-tune, the eight-student file (W8) scored 0.394486 on public, 0.001032 above the best click-label file. An audit of the distillation path found no bug and no leak.

### Width 192

Width 192 became useful only here. A pool of n students of one recipe follows the pool-size law NDCG(n) = a - b / n ([Pool size](#pool-size)). Under distillation its asymptote a lies 0.00043 ± 0.00053 / 0.00090 ± 0.00031 higher at width 192 than at width 128.

By the rule written before the runs, nothing was promoted: the width-192 pool missed a +0.0005 validation bar by 0.00001. The eight width-192 students were pooled after that was seen. The choice was recorded as post hoc before any fine-tune number existed, and the public score bore it out.

Width 224 and 256, a third block, a second history layer and a feed-forward width of 4d added nothing. Width 224 read +0.00026 ± 0.00058 / +0.00016 ± 0.00032 against the same 4 seeds at width 192, and +0.00031 ± 0.00057 on validation at the last checkpoint. As all-data files on 18 November it read +0.00005 to +0.00017 on validation against FD16P20 and was not scored ([07](07-what-failed.md)).

### Teacher weight

The teacher weight of 0.5 is where the pool is best. With more teacher a single student gets better and the pool gets worse, because students that follow the teacher more closely are more alike.

| Teacher weight | 0.3 | 0.5 | 0.8 | 1.0 |
|---|---|---|---|---|
| Within-feed correlation between two students | 0.966 | 0.978 | 0.990 | 0.995 |
| Four-student pool, validation (width 128) | 0.39977 | 0.40085 | 0.40004 | 0.39906 |

At width 192 and 8 seeds, weight 0.8 gave the member +0.00126 ± 0.00033 / +0.00078 ± 0.00019 and the pool -0.00049 ± 0.00048 / -0.00029 ± 0.00026.

### Choice of teacher

What the teacher is made of did not matter in the range that could be built. Four students of the 25-member teacher, pooled, read 0.40095 / 0.32317, against 0.40085 / 0.32349 for four of the 11-member one (width 128). A teacher made of students and a cross-fitted teacher were no better. Two students of different teachers correlate like two students of one teacher (0.9718 against 0.9722). The final files pool students of both.

The cross-fitted teacher was tried because teacher scores are in-sample: the teacher reads 0.3465 NDCG on feeds its members trained on and 0.3237 on the holdout. Each train feed was scored by three models that never saw its user. Four students of it read 0.40079 / 0.32347 against 0.40134 / 0.32408 for the same seeds of the reference recipe (width 192, 11-member teacher).

That seed-paired read may be too low by up to 0.0006, because the reference's seeds 951 to 954 are a lucky draw that a student of another teacher need not share ([04](04-validation.md#seeds-shared-by-arm-and-reference)). Against the mean of the 70 possible 4-seed reference pools, at the last checkpoint, the cross-fitted teacher reads -0.00015 ± 0.00047 on validation and -0.00059 ± 0.00026 on the holdout: level on validation, lower on the holdout.

Little was there to remove. The teacher's softmax is nearly flat (entropy 4.41 of 4.90 nats, 0.032 of its mass on clicked items), and the in-sample copy of the label is 0.2 to 0.7% of the target mass. Cross-fitting mainly weakened the teacher (holdout 0.32100 against 0.32370), and the idea was closed ([experiments/distillation](../experiments/distillation/)).

### Pool size

Pooling more students stopped helping because students of one target are near copies ([experiments/pool-size-and-growth](../experiments/pool-size-and-growth/)). Two click-label members correlate at 0.949 within a feed, two full-view students of width 192 at 0.973.

Every subset of the eight held-out full-view students fits NDCG(n) = a - b / n to 1e-4, with a = 0.40181 / 0.32472. Eight students are already within 0.00024 ± 0.00012 / 0.00017 ± 0.00007 of a. Sixteen students against eight read -0.00007 ± 0.00034 on validation and -0.000003 on public.

Files with 32 and 48 all-data full-view students read 0.40237 and 0.40222 on validation, below the 0.40271 of FD16P20, whose 16 are a favourable draw ([07-what-failed.md](07-what-failed.md#18-growing-the-ensemble-past-16-members)). On public, files with 24 and 48 full-view students scored 0.394905 and 0.39484072, level with the 0.394892 of FD16P20. A 68-member file scored 0.3946427.

The seed is the only source of difference left between students of one view. The one addition that still moved the pool was the second input view.

Two variants of how the final students were trained are told with the final files: [the pseudo-feed students of the second pick](#the-pseudo-feed-students-of-the-second-pick) and [members trained on all data](#members-trained-on-all-data).

## Fine-tune on the validation days

Validation is the only labelled data that looks like test: random order, and the days nearest to it. Each member is trained a little further on it, and the pool keeps both versions of the member, the train-only branch and the fine-tuned branch ([experiments/fine-tune](../experiments/fine-tune/)).

### How the fine-tune runs

The fine-tune (`avito_ranker/training/finetune.py`; lab original `lab/nn_x/w5/finetune.py`) starts from the kept checkpoint. It splits the 14,423 validation feeds into 5 folds by user, so that no user is on both sides of a fold.

| Setting | Value |
|---|---|
| Fold groups | users; each of the 658 feeds of user 0 is its own group, because user 0 is a bucket of unrelated sessions |
| Runs | 3 seeds per fold, 4 epochs each |
| Optimiser | AdamW, learning rate 5e-5, weight decay 0.02, 30 warmup steps, cosine decay to zero |
| Batches | at most 8,192 candidates and 128 feeds |
| Loss | the listwise loss with clicks alone |
| Weights that predict | an exponential moving average (EMA) of the weights, 0.99 from epoch 0.5, taken at the last step; no epoch is selected |
| Prediction for a validation row | the 3 models that did not train on its fold (out of fold) |
| Prediction for a test row | the mean of all 15 models |

### What changes in the model

The domain embedding switches to its third row. And 42 extra inputs enter through a new branch, an MLP with layers of 42, 384 and 192 units, whose output is added to the candidate token. Its last layer starts at zero, so the fine-tune begins exactly at the train-only model. The branch trains at 10 times the base learning rate.

| Extra inputs | Count |
|---|---|
| The candidate's centrality in its feed | 1 |
| Leave-user-out counts with their within-feed rank, z-score and gap, and for all ten event types | 14 |
| Co-visitation and history-similarity columns | 6 |
| The near-duplicate family | 10 |
| Scores of the trajectory encoders trained on the full corpus | 11 |

The last group cannot be built for train rows: those encoders saw each train user's latest snapshot, which holds the positives of the user's earlier feeds.

### What the fine-tune is worth

On one early model the fine-tune added 0.0040 out of fold (0.38068 train-only, 0.38471 fine-tuned), most of it through the extras: with a smaller set of 27 item-level inputs it reached 0.38123. Without any extra input it re-calibrates the model toward cold items: +0.0099 ± 0.0017 on validation feeds whose positives are all cold and -0.0032 ± 0.0008 on feeds whose positives are all warm.

As the train stage took over such inputs, the validation gain shrank, and in the final recipe the fine-tuned branch looks worthless on validation. Out of fold it reads 0.40190 against 0.40213 for the train-only branch of FD16P20. For the eight-student pool the 0.5 / 0.5 mix is +0.00009 ± 0.00043 above the train-only branch, where public put it at +0.00048.

The 0.0004 between those two reads was first put down to the folds (3 models score a validation row, 15 score a test row). A later check measured that term at +0.00006 to +0.00008.

The public-mix view reads the branch at +0.0005 to +0.0007, with every cell within 1.6 standard errors of zero ([04-validation.md](04-validation.md#the-witnesses)). The branch reads positive on feeds with many cold candidates, which public weights more heavily than validation does.

On public the eight-student pool was submitted at three weights of the fine-tuned branch:

| Weight of the fine-tuned branch | 0 | 0.5 | 1 |
|---|---|---|---|
| Public NDCG | 0.394009 | 0.394486 | 0.393960 |

The mix is worth about 0.0005 over either branch (+0.00048 ± 0.00055 over train-only), and the best click-label pool had shown +0.00047. A parabola through the three points peaks at 0.49. The weight stayed at 0.5.

### Changes tested and not adopted

| Change to the fine-tune | Result |
|---|---|
| A refit on all of validation without folds for the test predictions | -0.00025 ± 0.00018 in a nested simulation |
| Teacher targets in the fine-tune | -0.00036 ± 0.00031 and +0.00002 ± 0.00044 in two pilots |
| Feed weights matched to the private mix | -0.00007 ± 0.00022 |
| A 35-trial search over the fine-tune settings | +0.00055 ± 0.00041 on its confirmation set, below its bar |
| Weight 0.67 for the fine-tuned branch | +0.00024 ± 0.00025 nested over 5 user folds (19 members); -0.0003 ± 0.0002 on a 6-member family |
| Weight 0.60 for the fine-tuned branch | -0.00011 ± 0.00016 |

Three seeds per fold in place of one changed nothing on public (0.392754 against 0.392820 for the best click-label pool); three were kept.

## Ensemble

![Member predictions are z-scored within the feed and averaged into a pool of 16 full-view students and a pool of 4 popularity-light students, separately for the fine-tuned and the train-only predictions; in each branch the popularity-light pool enters at weight w and the full-view pool at 1 - w, with w = 0.4 on feeds of fewer than 50 candidates and 0.2 on the others in pick 1 (0.2 on every feed in FDX16P20:ps, FD16P20 and W8P20); the fine-tuned branch (validation 0.40139) and the train-only branch (0.40208) are mixed 0.5 / 0.5 into the final score (0.40293), which is ranked within each feed and written to a file of 2,904,806 rows](figures/ensemble.svg)

```text
z(x)      = (x - mean of x in the feed) / (std of x in the feed + 1e-9)
pool_b(S) = mean over the members m of S of  z(prediction of m in branch b)        b = fine-tuned, train-only
w         = 0.4 if the feed has fewer than 50 candidates, else 0.2                first pick; second pick, FD16P20, W8P20: 0.2 on every feed
B_b       = (1 - w) * z(pool_b(full-view students)) + w * z(pool_b(popularity-light students))
final     = 0.5 * z(B_fine-tuned) + 0.5 * z(B_train-only)
score     = rank of final within the feed, 0 = lowest
```

Members are z-scored within the feed because their scores live on different scales and only the order inside a feed counts. The z-mean ties with a mean of raw logits (+0.00004 ± 0.00015 on validation), so it is a convenience with no score claimed for it.

Nothing in the formula is fitted by an optimiser. Members have equal weight inside a pool. The popularity-light weights and the 0.5 / 0.5 mix are constants in the recipes of `configs/ensembles/`, applied by `avito_ranker/ensemble/`. The 0.4 on short feeds is the one constant read off a scan of validation scores ([The two input views](#the-two-input-views)). The figure shows the first pick.

Fitted combiners lost each time they met the board ([04-validation.md](04-validation.md#noise); [experiments/hard-label-pools](../experiments/hard-label-pools/)). A stacker that read +0.00056 on validation scored 0.00035 below its pool on public. The pool picked for the best validation score (0.40094) scored 0.00039 below the plain pool.

### The submission file

The file holds the within-feed rank of the final score, one row for each of the 2,904,806 test candidates. A rank is exact in float32, and exact ties are broken by a fixed random number per row. The file therefore has no ties for the metric to resolve with the negatives first.

The z-scores are computed by a float32 kernel. An independent rebuild in plain numpy ordered 4 of the 28,595 test feeds of FD16P20 differently (near-ties), so a byte-identical file needs that kernel. The package keeps the kernel and the tie key, and both picks and the three earlier candidates rebuild byte for byte from the stored member predictions.

## The final files

| | FD16P20@m0.4_0.2 (first pick) | FDX16P20:ps (second pick) | FD16P20 (first pick until 2026-11-18) | W8P20 (second pick until 2026-11-19) |
|---|---|---|---|---|
| File the pipeline writes (not in this repository) | `cand_FD16P20_m04_02kd_r4.parquet` | `cand_FDX16P20_pskd_r4.parquet` | `cand_FD16P20kd_r4.parquet` | `cand_W8P20kd_r3.parquet` |
| Full-view students, width 192 | 16: seeds 951-958 for each teacher | 16 pseudo-feed students: seeds 951-958 for each teacher | those of the first pick | 8: seeds 951-954 for each teacher |
| Popularity-light students | 4: width 192, 11-member teacher | those of the first pick | those of the first pick | 4: width 128, 25-member teacher |
| Weight of the popularity-light pool | 0.4 on feeds of fewer than 50 candidates, 0.2 on the others | 0.2 | 0.2 | 0.2 |
| Train feeds per member | 3,151,208 | 3,151,208, and about 21.7 million pseudo-feeds per full-view student | 3,151,208 | 3,121,208 |
| Steps and kept checkpoint | 99,840, the last step | 99,840, the last step | 99,840, the last step | 98,350, chosen on the holdout (step 92,500, 97,500 or 98,350) |
| Validation NDCG: final, fine-tuned branch, train-only branch | 0.40293, 0.40139, 0.40208 | 0.40286, 0.40194, 0.40234 | 0.40271, 0.40190, 0.40213 | 0.40214, 0.40167, 0.40212 |
| Public NDCG | 0.3952092 | 0.39526061 | 0.394892 | 0.3950094 |
| sha256 | `3cf6faccbad1357ae1a1b11999fef8168835d77407684832c7578c832c6e71b3` | `8235c55cb35931129a845ce6825a94913f81bebb51824f84bced6eadfb1571db` | `14adc19ac794e3977cc0275a0c15d2c76e55c5dcdeec911dc47a8193128626a8` | `406e5705cef75d264118eeb20102ab963cedf6d00b0b7c6be7c2f1b1361a4d78` |

All four files passed the pre-submission gate with 0 failures and 0 warnings. The first pick is FD16P20 with another weight on short feeds, and the two files give the same order on all 17,434 test feeds of 50 or more candidates. The second pick shares their 4 popularity-light students.

The team made FD16P20@m0.4_0.2 the first pick on 2026-11-18, after its public score, and W8P20, the strongest file built from other runs, the second. For W8P20 the team dropped its rule that every final file is trained on all data.

On 2026-11-19 FDX16P20:ps replaced W8P20, again after its public score, and both picks now keep the rule. Both files are selected on the platform between 2026-12-08 21:00 and 2026-12-09 09:00 UTC, after the freeze on 2026-12-07.

[11-final-picks.md](11-final-picks.md) gives the reasons, the audit and the risks, and describes FD16P20vf, the other candidate for the second pick until 2026-11-18.

### The pseudo-feed students of the second pick

The 16 full-view students of the second pick follow the recipe of the first pick's students, with one addition. During the constant-rate phase, the first 79,872 steps, each batch draws one pseudo-feed per real feed, built on the GPU from that feed's own history (`lab/nn_x/w12/pseudokd/cd_pseudokd.py`). The 86% of draws that find a positive are kept. In each pseudo-feed:

- The positive is a card the user opened from an earlier feed (event type 1, source 1) among the 64 newest history events, of an item that occurs in no older event and is not a candidate of the feed.
- The history is the events older than the positive by at least g places, with g drawn uniformly from 16 to 64.
- The negatives are the feed's non-clicked candidates, minus those the user interacted with after that cut.
- The user embedding is left out, and f1 to f5 and the pre-computed inputs take their missing path.

The pseudo-feeds add 0.3 times their listwise loss, with a one-hot target and no auxiliary loss. A linear head on the candidate's content embedding and category, zero at the start, adds to their scores only, so the shared layers need not learn which items users opened in the past. The cooldown and the fine-tune use real feeds only, and the users of the 30,000 holdout feeds are never a source.

Each student saw about 21.7 million pseudo-feeds. The trainer is `lab/nn_x/w12/pseudokd/train_full_pk.py`, and the package trainer has no pseudo-feed option.

As an idea the pseudo-feeds failed. On held-out students of the 11-member teacher a 2-seed screen read +0.00092 ± 0.00060 per member, then the 8-seed pool +0.00054 ± 0.00047 on validation and +0.00010 ± 0.00027 on the holdout (randomization p 0.19), below its rule. All-data try files followed at the request of the person on the team who uploads the files ([13-how-the-work-was-run.md](13-how-the-work-was-run.md#how-the-work-was-run)):

| File | Full-view students | Validation, against FD16P20 | Public, against FD16P20 |
|---|---|---|---|
| FDXH16P20:ps | the 8 of the 11-member teacher replaced by pseudo-feed students | 0.40298, +0.00028 ± 0.00029 | 0.3947873, -0.00010 |
| FDX16P20:ps, the second pick | all 16 replaced | 0.40286, +0.00015 ± 0.00036 | 0.39526061, +0.00037 |
| FDXU32P20:ps | the 16 pseudo-feed students added to the 16 of FD16P20 | 0.40301, +0.00029 ± 0.00026 | 0.39524613, +0.00035 |

FDX16P20:ps scored 0.00047 above FDXH16P20:ps on public, 1.3 times their paired standard error, after reading 0.00012 below it on validation. Neither validation nor public separates the pseudo-feed students from the plain ones.

### Members trained on all data

The members of FD16P20, which are also those of the first pick, were trained on all data on principle ([experiments/full-data-retrains](../experiments/full-data-retrains/)). The 30,000 holdout feeds are 0.95% of the train feeds but 18.3% of the random-order pool, the train feeds that look most like test. The expected gain from them was +0.00005 to +0.0002 on validation, less than can be measured, and public showed no gain (FD16P20 minus W8P20: +0.00058 ± 0.00043 on validation, -0.00012 ± 0.00056 on public).

An all-data member has no feed left to choose a checkpoint on and no out-of-sample score. Each one was accepted against its held-out twin, the run with the same command and seed, by three checks:

1. The log shows 3,151,208 feeds and the last step kept. The published read-out of each member, `<member>_fulldata.json`, gives both as `n_training_feeds` and `kept_step`; it sits in `lab/nn_x/w5/logs/` for full-view students and in `lab/nn_x/abl/logs/` for the others.
2. The in-sample holdout score exceeds the twin's out-of-sample score by at least 0.00428 (observed: +0.0158 to +0.0179).
3. The pool of the all-data members matches the pool of their twins on validation. For the 16 full-view students it read 0.40161 against 0.40157 (+0.00004 ± 0.00042); for the 4 popularity-light students +0.00001 ± 0.00059 on train-only predictions.

The four popularity-free students of FD16P20vf have no held-out twin. They passed a version of the check that needs none: structure of the log, and an in-sample holdout lift of +0.0153 to +0.0163.

No acceptance check is on record for the 16 pseudo-feed students of the second pick. Their logs show 3,151,208 feeds and the last step kept. For the 8 students of the 11-member teacher the held-out screen runs are twins, and the in-sample holdout lift over them is +0.0139 to +0.0155 (read from the logs on 2026-11-19). The 8 students of the 25-member teacher have no held-out twin.

## Costs

Every training run used one GPU. The runs had a few NVIDIA data-centre GPUs; the model and the number are not given here. The train split and its 34 inputs take 36.8 GiB of GPU memory, with a peak of 52.5 GiB while they load (the 21.2 GB in the lab logs is the split without the inputs).

Times are from the original runs unless marked. They are wall-clock per process on GPUs shared by several jobs, so they are upper bounds: 10-epoch runs at width 128 took 57 to 60 minutes in one batch of runs and 180 to 182 in another. Times marked "re-run" come from the package run of 2026-11-16 ([10-reproduce.md](10-reproduce.md)), one process per GPU.

| Stage | Runs | Hardware | Time per run | GPU-hours |
|---|---|---|---|---|
| Flatten | 3 splits | CPU | 16 s (train) | |
| Exploration probability | 1 | CPU | not recorded; 11 min in the re-run | |
| Leave-user-out counts | 1 | CPU | 44 s | |
| Co-visitation, train rows | 1 | CPU | 31 min | |
| In-feed inputs | 3 builders | CPU | not recorded; 5 min on train in the re-run | |
| Trajectory encoders | 4 on the full corpus, 6 on windows | 1 GPU | 13 to 59 min | 5.2 |
| Members of the 11-member teacher | 11 (7 of them in two stages) | 1 GPU | 118 to 157 min plus 23 to 51 min, or 180 to 182 min | 30.9 |
| Further members of the 25-member teacher | 14 | 1 GPU | 180 to 209 min (GRU), 57 to 60 min (popularity-light) | 27.1 |
| Teacher scores on all train candidates | 10 dumps | 1 GPU | at most 4.6 min for a dump of 1 or 2 checkpoints, not recorded for the four dumps of 3 or 4; 105 s per member in the re-run, plus about 1 min per input view to load the train split | |
| Students of FD16P20 | 20 | 1 GPU | 221 to 259 min | 81.8 |
| Fine-tune | 20 | 1 GPU | 15 to 22 min | 6.6 |
| Twin check, pooling, file, gate | 1 | CPU | under a minute | |

A member of FD16P20 costs 4.4 GPU-hours: 4.1 to train and 0.3 to fine-tune. Its 20 members cost 88.4 GPU-hours, and the pipeline from raw data 151.6, not counting the teacher dumps and the held-out twins that the acceptance check compares against. The 12 members of W8P20 cost 40.2 GPU-hours (38.0 to train, 2.2 to fine-tune).

The 16 pseudo-feed students of the second pick trained in 109 to 115 min each, 29.7 GPU-hours in all, fine-tunes not counted. Each ran alone on a GPU, while the students of FD16P20 (221 to 259 min) shared each GPU with other jobs, so the two times are not comparable.

## Known limits

The board cannot separate the files of this design. Seventeen scored files span 0.39436 to 0.39526 on public, and each lies within 1.1 paired standard errors (0.00026 to 0.00056) of FD16P20 ([11-final-picks.md](11-final-picks.md#risks)). The two picks are among the top three of the seventeen, and the best of seventeen noisy scores over-states the level of its file.

Public is 8,562 test feeds; private is the other 20,033 and stays unknown until 2026-12-09. The places of the picks on the public board are in [05, Board snapshots](05-timeline.md#board-snapshots).

Larger backbones did not help. Beyond the widths, depths and head counts above, large language model (LLM) backbones of 0.6 to 9 billion parameters were 0.0038 to 0.0053 worse on validation. Routed experts and depth 7 for the students (19 November) did not help either. More data helped: training without 20% of the users cost a member 0.0028 / 0.0022.

[Section 6 of 12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) lists five ideas about the amount of labelled data and the cleanliness of the labels that were closed on paper and never had a fair run. The page gives them as a judgement, with no experiment behind it.

The popularity-light share rests on thin evidence. Validation and public together give z 2.0, the holdout reads it negative, and the lagged holdout, which should remove the holdout's bias, shows no gain either ([04-validation.md](04-validation.md#the-witnesses)). Private inputs are staler than public ones (11.23 days against 9.34), and every file carries that exposure. The short-feed weight of the first pick rests on less: one public reading and a validation curve with a single high point.

All-data members have no out-of-sample score. Of the 16 scored files with all-data members, 12 sit 0.00009 to 0.00097 below the line public = 0.0074 + 0.963 × validation, and 4 sit up to 0.00022 above it. W8P20 and two files of held-out full-view students alone (8 and 16 students) sit about 0.0003 above it ([04-validation.md](04-validation.md#noise)).

Among the all-data files the residual falls as validation rises. Part of the gap is therefore validation over-stating the files with a favourable member draw ([11-final-picks.md](11-final-picks.md#risks)).

A retrain is a new draw. GPU training is not bit-reproducible, the validation scores of the 16 full-view members have a standard deviation of 0.00048, and two independent 16-member pools of one recipe differ by 0.00030 rms (root mean square) on validation. The 0.40271 of FD16P20 is a favourable draw; the unbiased level of its recipe is about 0.4024. Prediction is reproducible: the 36 member checkpoints re-scored with the package reproduce their stored predictions exactly.

Validation and test take different paths through the fine-tuned branch. A validation row is scored by 3 models and a test row by 15. The encoder extras of validation come from encoders trained without test snapshots, those of test from encoders trained with them. In the train stage, 27.4% of train feeds (Sep 6-8) have every encoder input blank, a pattern that never occurs on validation or test.

Parts of the lineage did not survive as package code: the script that averaged the teacher files, the exact trainer source of the W8P20 students, and a trainer for the pseudo-feed students of the second pick, which need `lab/nn_x/w12/pseudokd/train_full_pk.py` ([10, Gaps in the lineage](10-reproduce.md#gaps-in-the-lineage)).

Never varied in a full run: the warmup length, Muon's momentum, the AdamW betas, a learning rate above 2e-3 at the normal batch size and the cooldown floor of 3%. A 40% cooldown and a learning rate of 4e-3 at 4 times the batch size did run, and neither was adopted. Never removed in a test: the linear skip from the dense block and the f1 to f5 columns.
