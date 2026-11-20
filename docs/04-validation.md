# Validation: telling a real change from noise

In brief:

- Test labels were visible only through the public board: 5 submissions a day, scored on 8,562 feeds. Every other decision rested on five samples, called witnesses, each blind or biased somewhere.
- Validation predicts the public score of a file to about 0.0006 (public = 0.0074 + 0.963 × validation). Neither validation nor the public board can order near copies, files with a within-feed correlation of 0.995 or more and a validation difference of at most 0.002.
- Rules were written down before the runs they judged, and from the evening of 2026-11-11 every file had to pass a pre-submission gate. Two rules in force were wrong, the holdout's veto on popularity changes and a bar set against one realised pool, and two older ones failed their purpose ([Rules written before the read](#rules-written-before-the-read)).

## Notation and files

- Scores are NDCG over the full candidate list, averaged over feeds. [glossary.md](glossary.md) defines this and the other terms, and lists the code names of the working records.
- A difference is paired per feed. "±" is one standard error (SE) of that paired difference, and z is the difference divided by it. A pair of numbers a / b is validation / holdout.
- A member is one trained ranker. A pool is the equal-weight mean of its members' within-feed z-scores. A file is a submission: half fine-tuned branch, half train-only branch ([03-solution.md](03-solution.md)).
- A student is a ranker trained by distillation: its target mixes the clicks with the averaged scores of a teacher ensemble ([03-solution.md](03-solution.md#distillation)). A full-view student reads all 34 pre-computed inputs. A popularity-light student has the 8 popularity inputs (3 co-visitation scores, 5 leave-user-out counts) replaced by one clipped exposure count. Wide means width 192, narrow width 128.
- A near copy is a file with a within-feed correlation of at least 0.995 to its reference and a validation difference of at most 0.002.

The files named on this page:

| File | Members | Members trained on | Validation | Public |
|---|---|---|---|---|
| G3, the best click-label pool | 20 rankers trained on clicks alone | train feeds without the 30,000 holdout feeds | 0.40032 | 0.392820 |
| C3, the best click-label file | the 20 of G3 plus 8 popularity-light rankers | train feeds without the holdout feeds | 0.40101 | 0.393454 |
| W8 | 8 wide full-view students | train feeds without the holdout feeds | 0.40165 | 0.394486 |
| W16 | 16 wide full-view students | train feeds without the holdout feeds | 0.40158 | 0.39448337 |
| W8P20, pick 2 until 2026-11-19 | W8 plus 4 narrow popularity-light students at weight 0.2 | train feeds without the holdout feeds | 0.40214 | 0.3950094 |
| FD16P20 | 16 wide full-view students plus 4 wide popularity-light students at weight 0.2 | all train feeds | 0.40271 | 0.394892 |
| FD16P20@m0.4_0.2, pick 1 | the 20 members of FD16P20; popularity-light weight 0.4 on feeds of fewer than 50 candidates, 0.2 on the others | all train feeds | 0.40293 | 0.3952092 |
| FDX16P20:ps, pick 2 | 16 wide full-view students that also train on pseudo-feeds built from user histories, plus the 4 popularity-light students of FD16P20 at weight 0.2 | all train feeds | 0.40286 | 0.39526061 |

The scripts and read-outs behind this page are in [experiments/validation-and-leaks](../experiments/validation-and-leaks/) and [experiments/popularity-staleness](../experiments/popularity-staleness/).

## The witnesses

Apart from the public board, every decision rested on five samples. Validation and the holdout carry labels. The lagged holdout, the private-mix view and stale validation are derived from them by rebuilding inputs or re-weighting feeds. These five are the witnesses; the public board is not one of them. Each is blind or biased somewhere, and two of the team's worst mistakes came from trusting one of them alone.

| Sample | Feeds | Days (Sep 2026) | Feeds with fewer than 50 candidates | Candidates with a zero popularity count | Days since a warm item was last seen in train | Weekend feeds |
|---|---|---|---|---|---|---|
| Validation | 14,423 | 22, 24 | 40.9% | 38.1% | 5.69 | 0% |
| Holdout | 30,000 | 6-19 | 0% | 22.0% | 1.75 | 32% |
| Public | 8,562 | 26-27 | 38.7% | 42.9% | 9.34 | 100% |
| Private | 20,033 | 27-29 | 39.2% | 45.9% | 11.23 | 24% |

The 8 popularity inputs are counted on the train days, so their age grows with the distance from Sep 19. That age, in the last-but-one column, is why the witnesses disagree: the holdout sees those inputs 1.75 days old, private 11.23 days old. A warm item is one that was a train candidate before the feed's day.

### Validation

Validation is 14,423 feeds (1,389,971 candidates) from two weekdays, shuffled before display like test. It reads levels and large steps correctly: between two day-1 files a validation gain of +0.00707 ± 0.00108 became +0.00766 on public. It cannot order near copies ([Noise](#noise)), and its popularity inputs are fresher than those of test.

Validation reads the fine-tuned branch out of fold, and reads it low. For W8, removing the branch cost 0.00009 ± 0.00043 on validation and 0.00048 ± 0.00055 on public ([03-solution.md](03-solution.md#fine-tune-on-the-validation-days) has the branch, the folds and the rest of the evidence). The public-mix view (below) reads the branch at +0.0005 to +0.0007, close to public, with every cell within 1.6 standard errors of zero.

The branch reads positive on feeds where 45% or more of the candidates are cold. Those feeds make up 21% of public, 33% of private and 8% of validation. No cause has been confirmed.

### The holdout

The holdout is 30,000 train feeds, drawn once with a fixed seed from the feeds that the team's detector marks as random-order with probability above 0.9 and that have at least 50 candidates ([02-what-the-data-told-us.md](02-what-the-data-told-us.md#random-order-feeds-inside-train)). Members trained without these feeds are called held-out members. Checkpoints and members were chosen on the holdout, so that validation was not the sample selected on. For train-only members it is a second witness, with twice the feeds of validation.

The holdout is not free of selection: it chose every checkpoint and the composition of the best click-label pool.

In the click-label period four pools had a public score and a train-only holdout score to compare with G3. Taking each holdout difference as the public difference, with no fitted parameter, fits these four contrasts at chi-square 0.27 on 4 degrees of freedom. The null model, all seven scored click-label pools equal on public, fits the board too (chi-square 2.9 on 6 contrasts), and the board cannot tell the two apart.

The holdout is blind or biased in three places:

- It has no feed of fewer than 50 candidates; such feeds are 39% of private.
- It has no read of the fine-tuned branch, which is read only on validation, out of fold.
- Its popularity inputs are biased. On its rows 55.6% of the popularity count mass comes from other users' snapshots taken after the feed (1.9% on validation), so it over-rewards anything that leans on popularity. Removing the popularity inputs cost 0.0074 on the holdout and 0.0023 on validation (2-epoch paired runs).

### The lagged holdout

The lagged holdout is the holdout with the 8 popularity inputs rebuilt at the lag that test rows have. A label-free check decides where it may be read: the distribution of the rebuilt inputs by listing age and coverage must match the test rows. The check failed on the full holdout for one of five criteria and held on the 13,424 feeds from Sep 13 on. A second build, for the input test of 2026-11-16, has 15,225 feeds.

On those feeds the lag cost W8's pool 0.0033 ± 0.0008 and the popularity-light pool 0.00049 ± 0.00038 ([02-what-the-data-told-us.md](02-what-the-data-told-us.md#cold-items-and-stale-popularity) has the staleness measurements). A 0.2 admixture of popularity-light students read -0.00050 ± 0.00032 there with the stored inputs and +0.00024 ± 0.00031 under lag. That second number was the most favourable of nine member compositions (mean +0.00003 ± 0.00023), so the lagged holdout was used neither as a veto nor as a confirmation.

### The private-mix view

The private-mix view (`pmix` in the records) re-weights the 14,423 validation feeds, in 6 feed-size bins by 4 cold-share bins, until they match the private part of test. Re-weighted to the public part instead, it is the public-mix view. It uses no test label.

It corrects composition and nothing else: the inputs of a re-weighted feed are still 5.69 days old. Its level is about 0.0045 too high, so its only role was a veto: a change that read below -1 standard error on the private-mix view was not adopted. It reuses the validation labels and cannot confirm a validation result independently.

### Stale validation

Stale validation re-predicts the same 14,423 feeds after the popularity inputs are replaced by the values they would have at private-like staleness. Three scenarios exist. The third was declared the faithful one before its first number, because the first two feed the model input patterns that occur on no test row. Its limits:

- It overshoots: 48.6% of its candidates have a zero popularity count, against 45.9% on private.
- CPU (float32) and GPU (bf16) predictions of one member differ by up to 0.0004 NDCG, so a comparison has to stay on one of the two paths.
- A pass at pool level did not carry to the file. A cooldown (the last 20% of training steps) on a stale view of the inputs read +0.00103 to +0.00120 for the 16-member pool and -0.00047 ± 0.00035 for the file built from those members ([07-what-failed.md](07-what-failed.md#19-a-cooldown-on-stale-popularity-inputs)).

### Which witness predicted public

Six pairs of scored files show which witness predicted public. Holdout values are for the train-only pools.

| Pair (second file is the reference) | Validation | Public-mix view | Holdout | Public |
|---|---|---|---|---|
| W16, W8 | -0.00007 | +0.00014 | +0.00004 | -0.000003 |
| W8 with 4 narrow popularity-light students at 0.2 (W8P20), W8 | +0.00049 | +0.00063 | -0.00063 | +0.000523 |
| train-only branch of W8, W8 with both branches | -0.00009 | -0.00066 | same pool | -0.000477 |
| W8 plus 8 narrow full-view students, W8 | +0.00008 | -0.00022 | -0.00028 | -0.000408 |
| best click-label file (C3), best click-label pool (G3) | +0.00069 | +0.00108 | -0.00073 | +0.000634 |
| W8, best click-label file (C3) | +0.00064 | +0.00047 | +0.00150 | +0.001032 |

- The public-mix view has the public sign in the five pairs where public moved. On the first pair public is a tie (-0.000003). Its slope through the origin is 0.83.
- Plain validation has the public sign in 5 of 6.
- For train-only pools the public-mix view over-read the size of the difference 1.7 times.
- The holdout has the wrong sign on both pairs that change how popularity enters.

### Later pairs of near copies

Later pairs were near copies, and on them validation and the public-mix view no longer predicted the sign. Two files scored on 2026-11-15 read positive on both and negative on public, inside its noise:

- -0.00015 for a feed-size rule that raised the popularity-light weight to 0.4 on feeds of 50 or more candidates (FD16P20@m0.2_0.4; validation +0.00026 ± 0.00021);
- -0.00012 for FD16P20 against W8P20.

Of the ten files scored from 2026-11-17 to 2026-11-19, each read against the file it was built from, validation had the public sign in six, and no public difference exceeded 0.00037. Five of them:

| Change | Reference | Validation | Public-mix view | Public |
|---|---|---|---|---|
| A1: on feeds of 50 or more candidates, the popularity-light part of the fine-tuned branch taken from the popularity-light students' train-only predictions | FD16P20@m0.4_0.2 | +0.00017 ± 0.00014 | +0.00029 | -0.00017 |
| Pseudo-feed students (trained also on lists built from user histories) in place of half the full-view students | FD16P20 | +0.00028 ± 0.00029 | +0.00031 | -0.00010 |
| Weight 0.4 on feeds of fewer than 50 candidates and 0.2 on the others: FD16P20@m0.4_0.2 | FD16P20 | +0.00022 ± 0.00024 | +0.00027 | +0.00032 |
| Pseudo-feed students in place of all 16 full-view students: FDX16P20:ps | FD16P20 | +0.00015 ± 0.00036 | +0.00001 | +0.00037 |
| Cooldown of the 16 full-view students re-run on stale popularity inputs: FDX16P20:hz | FD16P20 | -0.00047 ± 0.00035 | -0.00072 | +0.00008 |

A1 was the second change to the popularity-light part on long feeds to land about 0.0004 below its validation reading. The reverse form of the feed-size rule, in the third row, leaves every long feed as it was. Its public gain is about one paired standard error, and it became pick 1 ([experiments/mixing-and-segments](../experiments/mixing-and-segments/)).

The two pseudo-feed rows differ by about 0.0005 on public, inside the paired noise, in the order opposite to validation. The second became pick 2 ([experiments/distillation](../experiments/distillation/)). The stale-view file read -0.00047 on validation and +0.00008 on public, at a paired standard error of 0.00045. Public read it about 0.0005 better, a weak sign, inside the noise, that public inputs behave stale.

## Noise

### The paired standard error

The paired standard error is honest:

- Of the 13,740 logged-in validation users, 13,715 have exactly one feed, so clustering by user changes the error by a factor of 1.000.
- Fourteen pairs of runs with the same recipe and different seeds gave a root-mean-square z of 1.04, so the per-feed error also covers seed noise for train-only members.
- An early check of 11 November, on 3 pairs of runs, put the holdout's run-to-run spread at 2.3 times its per-feed error. Fifteen seed pairs of the main ranker did not reproduce it (root-mean-square z 0.83 on validation, 0.89 on the holdout), so the per-feed error was kept for both witnesses.
- Fine-tuned members are noisier: the spread between fine-tunes of the same recipe was 1.4 times the per-feed error.

### Smallest detectable effect

Smallest effect each sample can show at 80% power (2.8 standard errors; the arithmetic is [worked below](#the-noise-arithmetic-worked)):

| Comparison | Validation | Holdout | Public | Private |
|---|---|---|---|---|
| one member against one member | 0.0029 | | | |
| near-copy pool against near-copy pool | 0.00086 | 0.00060 | 0.00112 | 0.00073 |

A true gain of +0.0005 between near copies is below the 0.00112 that public can show, so a single public read of such a gain is mostly noise.

For the three derived witnesses the records hold standard errors of pool differences and no power calculation: about 0.0003 on the lagged holdout, 0.0002 to 0.0005 on the private-mix view and 0.0003 to 0.0006 on stale validation. Multiplied by 2.8 they give detectable effects of 0.0006 to 0.0017.

Apart from history attention (attention from each candidate to the user's history events, [03-solution.md](03-solution.md#the-ranker)), the single-run gains of the first day (+0.0020 to +0.0032 on validation) were at or near the 0.0029 that one member against another can show. They were confirmed only later, by the 2-epoch removal runs of 2026-11-13.

### Decision statistics

From 2026-11-14 a pool claim was read with a pooled statistic:

pooled z = (z on validation + z on the holdout) / sqrt(2)

A claim passed only with both differences non-negative and pooled z at least 2. The rule was calibrated on 400 random splits of 16 same-recipe students into 8 against 8. The null spread of pooled z is 0.80; the rule passes 0.25% of nulls and a true +0.0005 about 56% of the time. A pooled z between 1.6 and 2 bought four fresh seeds and a second read.

When two pools differ in members, 49% to 75% of the per-feed variance of their difference is re-drawn by changing one member. Claims about the make-up of a pool were therefore read against the mean of 200 random same-size subsets of all finished students of its recipe. A change of recipe was read against the reference recipe trained with the same seeds, with that mean as a second reference ([Seeds shared by arm and reference](#seeds-shared-by-arm-and-reference)).

For the cooldown rounds of 2026-11-17 the per-feed z of a 16-student pool gave way to a randomization read. Arm and reference were swapped inside random subsets of the 16 matched pairs of runs, 2,000 times, and an arm was judged by the share of swaps that reached its pooled z. Across the eleven arms of those rounds the null standard deviation of the pooled z was 0.87 to 1.43, where an honest per-feed error gives 1.

### The level line

Validation maps to public on a line:

public = 0.0074 + 0.963 × validation

The residual standard deviation is 0.0006. The line was fitted on the 18 files scored by 2026-11-14 and held with the files of 2026-11-15; earlier fits on fewer files gave 0.0022 + 0.976, 0.0071 + 0.964 and 0.0078 + 0.962. The offset comes from feed size: public feeds are longer, and longer feeds score lower.

![Two scatter plots of public NDCG against validation NDCG for the 38 scored files with a clean validation score, with the line public = 0.0074 + 0.963 × validation. Files whose members held out the 30,000 feeds and files with all-data members are marked separately; 12 of the 16 files with all-data members lie below the line, by up to 0.0010. Both picks are below the line, pick 1 by 0.0002 and pick 2 by 0.0001.](figures/val-vs-public.svg)

The line predicts the level of a model generation, a change of 0.002 or more on validation ([glossary.md](glossary.md)). It does not place a file inside one.

Sixteen scored files have all-data members, and twelve of them sit below the line (residuals -0.00097 to +0.00022, mean about -0.0003). Three held-out files of full-view students (W8, W16 and W8P20) sit about 0.0003 above it. Part of this is known:

- The validation score of FD16P20 carries about 0.0002 to 0.0003 of favourable member draw.
- The four all-data files above the line are those whose validation reads about 0.0005 below FD16P20 while their public score is within 0.00014 of it: the two re-cooled probe files of 2026-11-17, the 48-student file and the stale-view file scored on 2026-11-19.
- Most of the sixteen contain most of the members of FD16P20, so they are closer to one draw than to sixteen.

On public FD16P20 is 0.00012 below W8P20 at a paired standard error of 0.00056, while FD16P20@m0.4_0.2 and FDX16P20:ps are 0.00020 and 0.00025 above it. A gain from training on all data is not shown.

### Selection on validation

Inside one generation validation did not order files. Seven click-label pools spanned 0.00198 on validation and 0.00074 on public, and the rank correlation of the two orders was +0.04. The fitted transfer factor of validation differences was 0.16 ± 0.34. Random half-splits of validation put a factor that low at probability 0.015, so it was not sampling noise.

The cause was selection. On 2026-11-12 the team submitted the two files that looked best on validation:

- The pool of only the 14 members whose cooldown was re-run with 2% random-order batches read 0.40094 on validation, against 0.40032 for G3, which the holdout rule had chosen. It scored 0.392434 on public against 0.392820.
- A small neural gate that re-weights the 20 members of G3 per feed read +0.00056 on validation and scored 0.392473, 0.00035 below the plain pool.

The two landed 0.0010 ± 0.0005 and 0.0009 ± 0.0004 below their forecast, and the pools never selected on validation landed 0.0005 to 0.0010 above. The expected excess of the best of K near copies at a standard error of 0.00039 is 0.00045 for K = 5 and 0.00087 for K = 50, the size observed. None of 40 re-weightings of validation explained the shortfall ([experiments/validation-and-leaks](../experiments/validation-and-leaks/)).

### Near copies on the public board

The public board could not rank the last files either. Before each submission the paired public standard error was computed from the validation per-feed differences scaled to 8,562 feeds, and the band in which nothing would change was written down.

By 2026-11-19 seventeen scored files of the final family (full-view students plus a popularity-light pool) spanned 0.39436 to 0.39526 on public, with a mean of 0.394864 and a standard deviation of 0.00025. That spread is below the paired error between two of them (0.0002 to 0.0006).

If the 17 had one true level, the best would be expected about 1.8 standard deviations (0.00044) above their mean. The top file, pick 2, sits 1.6 above it and pick 1 1.4. [11-final-picks.md](11-final-picks.md#risks) lists the 17 and draws them with their errors.

### The noise arithmetic, worked

Six short calculations produce most of the errors and thresholds of this section. The inputs are the feed counts of the samples and three per-feed standard deviations measured on validation: 0.25 for the NDCG of one file, 0.037 for the difference of two near-copy pools and 0.125 for that of two single members.

```
1  SE of one mean score         s / sqrt(n)
2  SE of a paired difference    s_d / sqrt(n), where s_d^2 = 2 s^2 (1 - r)
3  smallest detectable effect   (z for 5% two-sided + z for 80% power) × SE = (1.96 + 0.84) × SE = 2.8 × SE
4  SE on m feeds                SE on n feeds × sqrt(n / m)
5  excess of the best of K      SE × expected maximum of K standard normal draws
6  pooled z of two samples      (z on the first + z on the second) / sqrt(2)
```

1. One score. On the 14,423 validation feeds the mean NDCG of a file has a standard error of 0.0021.
2. A paired difference. Both files are scored on the same feeds, and a feed that is hard for one is hard for the other. Across feeds the NDCG values of two near-copy pools correlate at r = 0.989, so most of the spread cancels in the difference. Its standard error is 0.00031, against 0.00104 between two single members. This r is not the within-feed correlation of scores that defines a near copy.
3. The detectable effect. A difference is significant at the two-sided 5% level when it exceeds 1.96 standard errors. For that to happen in 80% of repetitions, the true effect must lie another 0.84 standard errors higher. The factor 2.8 gives 0.00086 for pools and 0.0029 for single members.
4. Another sample. At the same per-feed spread, an error scales with the square root of the feed count. Public has 8,562 feeds, so its errors are 1.298 times those of validation: 0.00040 for the paired difference and 0.00112 for the detectable effect. The holdout's detectable effect is 0.00060 and that of private 0.00073.
5. The best of K. If K files share one true level, the best of them reads above it by the expected maximum of K noise draws: 1.16 standard deviations of the noise for K = 5, 1.79 for K = 17 and 2.25 for K = 50. At a standard error of 0.00039 the best of five reads 0.00045 too high. The next sample does not repeat that excess.
6. Two samples. With no effect, each z on two independent samples has a standard deviation of 1 and their sum one of sqrt(2), so the division brings the pooled z back to 1. The per-member distillation reads ([A rule that passed](#a-rule-that-passed-distillation)), +0.00257 ± 0.00047 on validation and +0.00200 ± 0.00028 on the holdout, give z of 5.47 and 7.14 and a pooled z of 8.92.

The block below needs numpy and no data. It reproduces every figure above.

```
import numpy as np
from statistics import NormalDist

n_val, n_hold, n_pub, n_priv = 14423, 30000, 8562, 20033   # feeds per sample
s = 0.25                           # per-feed sd of the NDCG of one file
s_pool, s_member = 0.037, 0.125    # per-feed sd of a difference: two near-copy pools, two single members

se_one, se_pool, se_member = (v / np.sqrt(n_val) for v in (s, s_pool, s_member))   # steps 1 and 2
r = 1 - s_pool**2 / (2 * s**2)
print(f"1  SE of one mean score: {se_one:.4f}")
print(f"2  paired SE: pools {se_pool:.5f}, members {se_member:.5f}; r between two pools {r:.3f}")

q = NormalDist().inv_cdf                                                           # step 3
k = q(0.975) + q(0.80)
print(f"3  {q(0.975):.2f} + {q(0.80):.2f} = {k:.2f}; detectable on validation: pools {k * se_pool:.5f}, members {k * se_member:.4f}")

for name, m in (("holdout", n_hold), ("public", n_pub), ("private", n_priv)):      # step 4
    f = np.sqrt(n_val / m)
    print(f"4  {name:7s} x {f:.3f}: paired SE {se_pool * f:.5f}, detectable {k * se_pool * f:.5f}")

x, dx = np.linspace(-8, 8, 160_001, retstep=True)                                  # step 5
pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
cdf = np.cumsum(pdf) * dx
e_max = {K: np.sum(x * K * pdf * cdf**(K - 1)) * dx for K in (5, 17, 50)}          # density of the largest of K draws
print("5  expected best of K noise draws, in sd of the noise:", ", ".join(f"K={K}: {v:.2f}" for K, v in e_max.items()))
print(f"5  best of 5 at SE 0.00039: {0.00039 * e_max[5]:.5f}")
# the same by simulation: np.random.default_rng(0).standard_normal((200_000, 5)).max(axis=1).mean()

z_val, z_hold = 0.00257 / 0.00047, 0.00200 / 0.00028                               # step 6
print(f"6  z {z_val:.2f} and {z_hold:.2f}; pooled z {(z_val + z_hold) / np.sqrt(2):.2f}")
```

It prints:

```
1  SE of one mean score: 0.0021
2  paired SE: pools 0.00031, members 0.00104; r between two pools 0.989
3  1.96 + 0.84 = 2.80; detectable on validation: pools 0.00086, members 0.0029
4  holdout x 0.693: paired SE 0.00021, detectable 0.00060
4  public  x 1.298: paired SE 0.00040, detectable 0.00112
4  private x 0.849: paired SE 0.00026, detectable 0.00073
5  expected best of K noise draws, in sd of the noise: K=5: 1.16, K=17: 1.79, K=50: 2.25
5  best of 5 at SE 0.00039: 0.00045
6  z 5.47 and 7.14; pooled z 8.92
```

## Leak incidents

### The later-snapshot leak and the encoder scores

The first leak, found on 2026-11-11, is in the data itself: 92.1% of a feed's positives are in the same user's next history snapshot. It reached the team's files twice:

- As inputs: later-history flags, and ALS (alternating least squares) factors trained on pairs that included later snapshots, about 0.002 of the day-1 best validation score of 0.3869.
- On 2026-11-12 as a deliberate test: an overlay that pushes the leaked items to the top of G3 scored 0.392819 on public against 0.392820 for its base file.

[02-what-the-data-told-us.md](02-what-the-data-told-us.md#the-same-user-later-snapshot-leak) tells both in full and lists the rules that followed. The contaminated arrays and predictions went to quarantine folders that no loader may read. A review the same afternoon found that the neural loader's cache still held hard links to the quarantined files; the gate below checks for such links by inode.

The second incident of 2026-11-11 was in the encoder scores built for train rows. Validation fell by 0.005 to 0.009 while the holdout rose by 0.006 to 0.0085 at equal steps. The encoder's item table held no train candidates, and a missing value marked positives on train rows only ([07-what-failed.md](07-what-failed.md#2-encoder-scores-for-train-rows-first-build)).

The rule since then: a train row is built exactly like a validation or test row, and a new input is accepted only after its missingness and distribution on train rows match those rows.

That evening the new gate failed its first real file. Its leak probe read z 2.0 on the first multi-recipe pool: +0.0123 ± 0.0061 more gain on the 117 validation feeds whose user has a later feed than elsewhere. Same-recipe comparisons swung by -0.016 and +0.011 on those feeds (z -2.0 and +2.1), and a controlled pair that differed only in the new input read -0.0009 ± 0.0039.

The team judged it noise and moved the failure threshold to z above 3, with z between 2 and 3 a warning that requires the controlled pair. The threshold was changed after it had fired, and the status log of that day, a working record, says so.

### Later cases and audits

Three smaller cases followed between 2026-11-13 and 2026-11-15:

- A count over other users' histories that used later days read +0.00033 on validation and -0.00001 when rebuilt from strictly earlier days.
- A teacher that sees later snapshots read +0.037 ± 0.0025 on the feeds it covers, all of it from events after the first interaction with the candidate. That is a soft copy of the label, and it was never built into a teacher.
- CatBoost's native target statistics on a grouping value that is constant within a feed or a user leak through the within-feed prefix and were not used.

One parity gap stayed in the model: popularity counts on a train row include later days of other users. It is popularity and no label, and a causal rebuild lost 0.0018 to 0.0023 on validation in two LightGBM proxies. The inputs were kept as built, and the gap is treated as a source of bias in the holdout.

Audits of the distillation path and of the all-data training path found no leak. On the three candidate final files of 2026-11-16 (FD16P20, FD16P20vf and W8P20), an independent audit tested 59 input columns for a reaction to a later feed of the same user and found none ([11-final-picks.md](11-final-picks.md#what-the-audit-checked-and-found)). Pick 1, FD16P20@m0.4_0.2, has the same 20 members as FD16P20 and differs from it only in the pooling weight on short feeds.

Pick 2, FDX16P20:ps, was built on 2026-11-18, after that audit. Its 16 full-view students follow the all-data path of FD16P20. The pseudo-feeds they also train on are built only from the histories that the training feeds already carry.

## The pre-submission gate

From the evening of 2026-11-11 every file offered for submission had to pass one script: `scripts/09_check_submission.py`, which runs `avito_ranker/validation/gate.py` (the lab original is `lab/src/presub_gate.py`). A FAIL blocks the file and a WARN needs a written justification. The six checks, with the output for pick 1:

| Check | What it tests | Verdict | Pick 1 (FD16P20@m0.4_0.2) |
|---|---|---|---|
| A, file validity | The (feed_id, item_id) pairs equal those of the sample submission, none is duplicated, every score is finite and no feed has two equal scores. | any breach fails | passed |
| B, reproduction | The file is rebuilt from the member predictions and the declared weights. | fails if the smallest per-feed Spearman correlation between file and rebuild is below 0.9999 | 1.000000 |
| C, inputs | For each member the input lists logged at training and at fine-tune time are matched against banned names (all-snapshot history counts and later-history flags anywhere, the first build of the encoder scores as a training input) and against the inodes of the quarantine folder. | a banned name or a hard link fails; ALS factors, hour of day, train-candidate statistics and a missing log warn | 20 members audited: 34 inputs for each of the 16 full-view members, 27 for each of the 4 popularity-light members |
| D, validation difference and leak probe | The paired difference to a reference file, for all feeds and seven slices (fewer than 50 candidates, 50 or more, Tuesday, Thursday, not-logged-in users, feeds with and without a later feed of the same user). The leak probe is the gain on the 117 feeds with a later feed minus the gain on the others. | a slice below -2 SE warns; a probe above 3 SE fails, between 2 and 3 warns | +0.00079 ± 0.00045 against W8P20; probe +0.00613 ± 0.00469 |
| E, forecast | Public and private scores predicted from validation re-weighted by feed size, ± 0.003. | report only | public 0.3949, private 0.3975 |
| F, parity | The per-feed rank agreement between the file and the reference, on validation and on test. | warns if the test value is more than 0.02 below the validation value (a construction break on the test side) | 0.9928 on validation, 0.9931 on test |

Pick 1 passes with 0 FAIL and 0 WARN. Pick 2 passes the same way. Against W8P20 its check D reads +0.00072 ± 0.00045 and its leak probe +0.00308 ± 0.00365.

### Stored outputs and the freeze check

On both picks, on W8P20 and on the other two candidate files of 2026-11-16, the ported gate prints the stored gate output ([records/gate-outputs/](records/gate-outputs/)).

A separate integrity audit on 2026-11-15 checked seven candidate files, W8P20 among them; pick 1 did not exist yet. It covered the sha256 of each file, the presence of all member predictions and the all-data flag in each member's training log, set on the four all-data files and absent on the three held-out ones. All 35 of 35 cells passed ([`AUDIT_TABLE.txt`](../lab/nn_x/w12/freeze/out/AUDIT_TABLE.txt)).

On the freeze date, 2026-12-07, `scripts/freeze_check.py` compares each file's sha256 with the record and re-runs the gate, ahead of the platform's selection window of 2026-12-08 21:00 to 2026-12-09 09:00 UTC. A run on 2026-11-19 passed both final files, FD16P20@m0.4_0.2 and FDX16P20:ps. Each sha256 equals the record, and each gate text is identical to the stored one.

### Limits of the gate

- The leak probe has 117 feeds and an error of about 0.005, so it sees only a concentrated leak.
- Check B failed a valid file on 2026-11-16. Four test feeds each held one exact float32 tie, and the submission writer breaks a tie with a fixed random number while the gate ranked it by row order. A tie-aligned copy passed.
- The forecast of check E runs about 0.0005 low, so the level line of [Noise](#noise) is quoted instead.
- The log copies that check C read for the all-data members were byte prefixes of the complete logs, copied while the runs were training. The audited lines stand at the head of a log, no verdict changed, and the copies were replaced.

## Rules written before the read

### How rules were recorded

From the first day a readout rule went into the status log before the run it governed. From 2026-11-14 the rules went into one ledger, each block stamped with the output of `date -u` (hand-written times had once run ahead of the clock). An amendment says what it replaces and that it precedes the number it governs. A choice made after seeing a number is labelled post hoc.

Once, on 2026-11-16, three blocks of an input test were entered after its first runs had started. The ledger states it, and no prediction of that test had been opened. The ledger is a working record and is not published.

### A rule that passed: distillation

The rule was written at 2026-11-14 03:14 UTC, before the first run, and read at 05:18 UTC on 4 seeds:

| Criterion | Bar | Read |
|---|---|---|
| member, validation | at least +0.0010 | +0.00257 ± 0.00047 |
| member, holdout | not in the promotion rule | +0.00200 ± 0.00028 |
| 4-member pool, validation | at least +0.0005 | +0.00198 ± 0.00061 |
| 4-member pool, holdout | not negative | +0.00021 |

Distillation was promoted.

### A rule that failed: the popularity-free view

Four students trained without any popularity input, added to the popularity-light pool, gave a file above 70 of 70 control files, in which four more students of the existing view take their place. It read +0.00036 ± 0.00019 against the matched control, and 0.40286 against a control maximum of 0.40275.

The rule written at that point asked for a replication: four fresh seeds, above at least 63 of the 70 controls and no more than one standard error below FD16P20. The fresh seeds were above 16 of 70 and read -0.00024 ± 0.00018. The view was not adopted. The first-draw file (FD16P20vf) later scored 0.394630 on public, 0.00026 or one paired standard error below FD16P20.

### Disclosed overrides

The wide students missed their bar of +0.0005 on validation by 0.00001 (+0.00049, with +0.00059 ± 0.00032 on the holdout). Files were built from them anyway, on the pooled holdout read of both wide arms (+0.00107 ± 0.00031).

Those files were labelled post hoc in the ledger before their fine-tune, and the order of preference among them was fixed before any fine-tuned number existed. They became the base of the final family: W8 scored 0.394486 on public, +0.001032 over C3.

On 2026-11-17, after a message of the person on the team that the rules were too restrictive, the team began to score try files. These are files on all-data members, built from ideas whose validation mean was positive (including ideas that had missed their rule) and sent to the board for information.

One of them, FD16P20 with the popularity-light weight raised to 0.4 on feeds of fewer than 50 candidates, scored 0.3952092, then the team's best public score. On 2026-11-18 the team made it pick 1.

The rule for replacing pick 1 ([11-final-picks.md](11-final-picks.md#why-pick-1)) asked for a recipe change that had passed a rule written before its run or, for a near copy, a public score of 0.395992 or more. This file meets neither, and the rule was set aside.

Its lead over FD16P20 is +0.00022 ± 0.00024 on validation and +0.00032 on public, about one paired standard error. Both leads are the best of several tries. On validation the short-feed weight peaks at 0.4 alone (0.40266 at 0.3, 0.40293 at 0.4, 0.40251 at 0.5). On public the file is the best of seven scored on 2026-11-17 and 18. The expected difference between the two files on private is about +0.00005, a tie.

For pick 2 the team first waived the rule that every final file is trained on all data and took W8P20, the strongest file built from runs other than those of pick 1.

On 2026-11-19 the team took another try file instead. FDX16P20:ps has 16 full-view students that also train on pseudo-feeds, an idea that had failed its 8-seed rule (randomization p 0.19). It scored 0.39526061, the team's best public score and 0.00005 above pick 1. Both picks now meet the all-data rule.

A ledger rule of 2026-11-17, written before the pseudo-feed files existed, said that such a probe changes no pick while its public score is within 2 paired standard errors of FD16P20. FDX16P20:ps is 0.8 paired standard errors above FD16P20, and the change of pick 2 set that rule aside.

### Rejecting on the holdout alone

The rule "stop if the holdout is negative" closed a cooldown on stale inputs on 2026-11-12, at validation +0.00026 ± 0.00033 and stale validation +0.00107 ± 0.00036. The holdout also read -0.00073 and -0.00063 on the two popularity-light admixtures that public then rewarded with +0.000634 and +0.000523.

On 2026-11-15, after a message of the person on the team not to reject an idea on the holdout alone ([13-how-the-work-was-run.md](13-how-the-work-was-run.md#how-the-work-was-run)), the evidence policy was split by family:

- Changes to how popularity or recency inputs are used are judged on validation, stale validation, the private-mix view and the lagged holdout. The holdout is reported without a vote.
- Every other change keeps the two-witness rule: validation and the holdout.

An audit of the 277 rows that the idea list held on 2026-11-15 found 38 decisions that had leaned on the holdout:

- 31 closures in which a negative holdout decided while validation was not negative or not read;
- 4 closed as a cold-for-warm trade on fresh validation;
- one rule of process;
- two that were not holdout vetoes after all.

15 of the 38 are in the popularity family. Three ideas were re-opened, and nothing outside that family changed status. The ledger entry, written before the audit was complete, says 33 closures. The catalogue in [07-what-failed.md](07-what-failed.md), compiled later, has 312 entries.

The admixture itself reached FD16P20 through this policy, which was written after the public read. The rule in force at the read, written before the file was scored, would have put the design into pick 1 only at a public gain of two standard errors (+0.0008) or more. For a smaller gain, down to -0.0006, it left the design a candidate for pick 2 at most. The gain was +0.00052, 1.3 standard errors.

The admixture's evidence is validation +0.00049 ± 0.00031 and two public reads of +0.00052 and +0.00041, against the holdout's -0.00063 ± 0.00022. The second public read compares FD16P20 with W16, which was trained without the holdout feeds, so that pair differs both in the admixture and in the training data. Both final files carry the admixture; if the staleness explanation is wrong, each loses about 0.0005.

### A bar set against a lucky draw

Larger pools had to stay within one standard error of the 16-student file on validation. The 32- and 48-student files failed (-0.00034 ± 0.00025 and -0.00049 ± 0.00028), although their members were exchangeable with the old ones. They failed because the 16-student file reads 0.40271, where the unbiased level of its family is about 0.4024 ([07-what-failed.md](07-what-failed.md#18-growing-the-ensemble-past-16-members) has the growth runs and two earlier references that were draws too).

Scored on 2026-11-17, the 24- and 48-student files read 0.394905 and 0.39484072 on public, +0.00001 and -0.00005 against FD16P20. A non-inferiority bar against one realised pool fails by noise more often than its nominal rate. The reference for a pool is the mean of same-size subsets ([Noise](#noise)).

### Seeds shared by arm and reference

Student screens from 14 to 19 November paired each arm's seeds 951 to 954 with the same 4 seeds of the reference recipe. That reference pool reads 0.40134 on validation, 3rd of the 70 possible 4-seed pools of the 8 stored references and 0.00057 above their mean. On the holdout it is 37th, at the mean.

On 19 November two arms were re-read against the mean of the 70 pools, the cure of the previous section. Their validation reads at the last checkpoint rose from +0.00009 and -0.00016 to +0.00070 and +0.00045. At 8 fresh seeds they read +0.00006 and -0.00001.

The cure did not apply, because the luck sits in the seeds. On validation, seeds 951 to 954 read above seeds 955 to 958 by +0.00113 for the reference and by +0.00082 and +0.00085 for the two arms, which share its teacher. A seed-paired read cancels most of that. The mean of the subsets does not, and counts the arm's share of the luck as a gain.

A recipe with another teacher, the 25-member one, did not share the luck (-0.00051 on the same comparison). An arm with another teacher may therefore read up to 0.0006 too low against a seed-paired reference. The cross-fitted teacher is one: at the holdout-chosen checkpoints it reads -0.00055 on validation paired by seed and +0.00002 against the mean pool, and about -0.0006 on the holdout either way.

The mean of same-size subsets is the reference when arm and reference share no seeds, as with the 32- and 48-student files above. With shared seeds and a shared teacher, the seed-paired read is better. With shared seeds and another teacher neither is clean, and only fresh seeds under a rule written first settle the question ([experiments/optimiser-and-modern-tricks](../experiments/optimiser-and-modern-tricks/), `lab/nn_x/w12/moe/logs/seedluck.txt`).

[Section 6 of 12-why-the-score-stopped.md](12-why-the-score-stopped.md#6-ideas-the-search-may-have-missed) asks for this read, with 8 fresh seeds, for five ideas that never had a fair run. [13-how-the-work-was-run.md](13-how-the-work-was-run.md#the-limits-such-a-setup-would-still-meet) shows what the limits of this page mean for a search that runs day and night.

### Older rules that failed their purpose

- On the first day a change was accepted if either witness reached 2 standard errors. At the measured correlation of the two samples that passes 1.6 times the nominal rate of nulls, and one input entered three members at validation z 1.44 on one seed.
- One rule could not return a verdict. A replay test of longer histories assumed about 11,600 qualifying holdout feeds and found 3,235, on which even the stored events read z 1.58. It ended undecided. Since then the number of qualifying feeds is counted before a rule is written.

## What no witness saw

No private score exists, and no labelled sample is as stale as private: its popularity inputs are about two days older than those of public. The projection private = public + 0.0026 comes from the feed-size mix alone.

Weekend effects cannot be checked either. Public is all weekend, private is 24% weekend, validation has none, and the holdout (32%) is the only labelled sample with both kinds of day.

The all-data members have no out-of-sample read. Those of FD16P20, and so of pick 1, were accepted against their held-out twins (the same command and seed, without the 30,000 feeds) by the check described in [03-solution.md](03-solution.md#the-final-files).

For the 16 pseudo-feed students of pick 2 no twin check is on record. Eight of them have held-out twins, the 8 seeds of the pseudo-feed test of 2026-11-17, and the file passed the gate.
