# Error lens: SEGMENTS

Where is the remaining loss (1 − NDCG per feed) of `nn_final2_val_oof` (NN oof) and `nn_final2_trainonly` (NN tr)? Where do the other models do better or worse? Which segments could a specialized treatment improve?

Scripts and logs: `<repo>/eda/errors/segments/`. All numbers are on val with the leak mask applied: 14,306 feeds, after removing the 117 feeds whose non-zero user has a later feed in val or test. Every feed in val has at least one positive. Comparisons give a paired per-feed delta with its SE in parentheses.

The tool is the positive-category oracle and its mirror, both built on NN oof:
- **cat-orc** ranks the categories correctly (positive categories first) and keeps NN order inside each category. It scores 0.5576, a gain of +0.1715 over 0.3861. Its gain measures **cross-category** headroom.
- **within-orc** keeps NN's category order (categories sorted by their best NN score) and puts the positives first inside each category. It scores 0.6169, a gain of +0.2308. Its gain measures **within-category** headroom.

Model columns:
- `stk` means `stk2_res2_r3`, the predictions this task assigned. That file was deleted while this analysis ran and replaced by the leak-free `stk2_res4_clean`. The per-segment tables use the old file. The follow-up tests (§5, §6) use `stk2_res4_clean`, labelled `stk_clean`.
- `blend` = 0.5·z(NN oof) + 0.5·z(stk), which is v5-like.

Compute: CPU only, at most 16 threads, total runtime under 5 minutes. No GPU was needed.

## 0. TL;DR

1. **Observable segments are not where the loss is.** Feed size explains loss level: the loss share of each M bucket ranges from 0.55× its feed share (M ≤ 10) to 1.20× (M > 200). Once feed size is controlled for (M ventile × npos cell), every observable segment sits within ±0.03 NDCG of expectation.
   - H, user group, train-seen, hour, day, cold share and number of categories all have a lift of 0.94-1.04.
   - **Routing a different blend weight by any observable segment loses in nested CV**: −0.0004 to −0.0012 against one global weight, for every pair tried (NN oof|stk, NN oof|CB, NN oof|NN tr) and for 10 segmentations. Segment-specific reweighting of the existing models has nothing left to give.
2. **The loss separates cleanly by two label-side properties.** Table §2.
   - **Intent regime.** In 52.5% of feeds the positive is in the user's f3-top category. Those feeds hold 48.6% of the loss and **88% of the within-category headroom**. In the other 47.5%, the "intent-switch" feeds, the positive sits in a lower-f3 category. They hold 51.4% of the loss and **85% of the cross-category headroom**.
   - **Positive coldness.** In 17.4% of feeds every positive is cold, i.e. never seen in train candidates or histories. Weighted to the test mix, that rises to **22.1%**.
3. **Val-label models win on warm items and lose on cold ones.**
   - Against NN tr, NN oof gains +0.0088 (0.0013) on warm-positive feeds and loses **−0.0159 (0.0020)** on all-cold-positive feeds. For `stk` the figures are +0.0101 and −0.0217; for `stk_clean`, +0.0092 and −0.0224.
   - The cause is missing item information, not bias. A constant cold offset gives NN oof −0.0002 (0.0001) and stk_clean +0.0004 (0.0003).
   - The weakest pair type for every model is a cold positive against a warm negative: 16% of pairs, AUC 0.659 for NN oof against 0.698 for warm vs warm, and 0.599 within the same category.
   - Weighted to the test (M × cold) mix, the NN oof − NN tr gain falls from +0.0034 to +0.0026.
4. **The available signals cannot reach the cross-category headroom.**
   - In intent-switch feeds, no category signal puts the positive category first more than 21% of the time. This covers f3, f1, NN max, CB, candidate share, and history counts by recency or event type.
   - A category-level LR on NN + f + history, fit in CV, lifts overall category top-1 from 0.513 to 0.570. As a candidate boost it nets 0: +0.0055 (0.0004) on f3-top feeds, −0.0060 (0.0006) on intent-switch feeds.
   - The regime can be predicted (AUC 0.772), but routing f3, CB or stk weights by the predicted regime gains 0.0000.
5. **Two more negative results.**
   - The same user's earlier train snapshots are allowed by the leak rules and add about 107 history items for 28.6% of feeds. Similarity features built from them give ±0.0003 (SE 0.0002-0.0003).
   - Simple priors on top of NN tr do not reproduce the val fine-tune gain: a category one-hot gives −0.0009 (0.0004), warmness flags +0.0009 (0.0006). NN oof gains +0.0034.
6. **Top 3 targets for specialized treatment, which need new information** (§7):
   - (1) within-category ranking in dominant-intent feeds, using item-to-item co-visitation or a within-category re-ranker;
   - (2) fresh-listing (cold) positives, using a fine-grained content-kNN quality prior and ID-dropout training;
   - (3) intent-switch feeds, using a cross-user category-transition model, with a strict go/no-go gate.

## 1. Totals

| model | val NDCG (clean) | test-mix (M) | test-mix (M × cold share) |
|---|---|---|---|
| NN oof | 0.3861 | 0.3798 | 0.3863 |
| NN tr | 0.3827 | 0.3764 | 0.3838 |
| stk (res2_r3) | 0.3855 | 0.3792 | 0.3870 |
| blend | 0.3867 | 0.3804 | 0.3871 |
| CB | 0.3695 | 0.3632 | 0.3688 |
| stk_clean (res4) | 0.3850 | | |

- Paired gains over NN tr on val: NN oof +0.0034 (0.0010), stk +0.0028 (0.0009), blend +0.0040 (0.0009).
- Under the M × cold mix (effective n = 7,155) the same gains are +0.0026, +0.0032 and +0.0034.
- Test feeds hold far more cold candidates than val: mean cold share is 0.395 in test against 0.290 in val. Feeds with ≥ 45% cold candidates make up 29.1% of test against 8.3% of val.
- Cold-heavy feeds are *easier* once M is controlled for: +0.0255 (0.0068).

## 2. Error map: intent regime × positive coldness

Headroom units: total loss (1 − NDCG) = 0.6139, cat-orc headroom = 0.1715, within-orc headroom = 0.2308.

| regime | positive | share | NN oof | loss share | share of cat-orc headroom | share of within-orc headroom | oof − tr | stk − tr | CB − oof | blend − oof |
|---|---|---|---|---|---|---|---|---|---|---|
| f3-top | warm | 34.9% | 0.4413 | 31.8% | 10.9% | 58.3% | +0.0114 (0.0021) | +0.0142 (0.0019) | **+0.0076 (0.0028)** | +0.0026 (0.0011) |
| f3-top | mixed | 8.3% | 0.4263 | 7.8% | 1.8% | 11.6% | +0.0007 (0.0023) | −0.0033 (0.0023) | −0.0224 (0.0033) | −0.0007 (0.0013) |
| f3-top | all cold | 9.3% | 0.3975 | 9.1% | 2.8% | 18.0% | **−0.0223 (0.0030)** | **−0.0283 (0.0032)** | −0.0400 (0.0046) | +0.0006 (0.0018) |
| **f3-top** | **all** | **52.5%** | 0.4311 | **48.6%** | 15.4% | **87.9%** | +0.0038 (0.0015) | +0.0039 (0.0014) | −0.0056 (0.0021) | +0.0017 (0.0008) |
| off-f3 | warm | 36.1% | 0.3395 | 38.8% | 66.9% | 7.8% | +0.0062 (0.0015) | +0.0061 (0.0014) | −0.0247 (0.0021) | +0.0005 (0.0008) |
| off-f3 | mixed | 3.3% | 0.3623 | 3.4% | 3.6% | 1.5% | −0.0027 (0.0033) | −0.0097 (0.0034) | −0.0287 (0.0055) | −0.0038 (0.0017) |
| off-f3 | all cold | 8.1% | 0.3118 | 9.1% | 14.0% | 2.8% | −0.0086 (0.0025) | −0.0142 (0.0022) | −0.0469 (0.0039) | −0.0042 (0.0012) |
| **off-f3** | **all** | **47.5%** | 0.3364 | **51.4%** | **84.6%** | 12.1% | +0.0031 (0.0012) | +0.0015 (0.0012) | −0.0287 (0.0018) | −0.0006 (0.0007) |

"f3-top" means the best positive's category has the highest f3 in the feed (category rank #1).

Readings:
- **Two error types, two regimes.** In f3-top feeds the category is right and the item choice inside it is wrong: cat-orc gap 0.050 per feed against within-orc gap 0.387. In intent-switch feeds the category is wrong: cat-orc gap 0.30 to 0.52 per feed against a within-orc gap of about 0.01-0.03.
- **CB beats NN oof in one cell only:** f3-top with warm positive, 34.9% of feeds, +0.0076 (2.7 SE). The stacker gains most there too. Item-level statistics from train carry within-category signal in that cell. The cell cannot be identified at inference, though: routing the CB weight by the predicted regime gives −0.0000 (§6).
- **All-cold cells are where the val-label components lose:** −0.009 to −0.028.

## 3. Observable segments (usable for routing)

Full tables, including the cat-orc and within-orc gaps for every bucket, are in `eda/errors/segments/report_tables.md` and `segments.csv`. Selected rows:

| segment | bucket | share val | share test | NN oof | loss share | M × npos-adjusted dev (SE) | oof − tr | stk − tr | CB − oof | blend − oof |
|---|---|---|---|---|---|---|---|---|---|---|
| M | 5-10 | 8.4% | 7.6% | 0.6593 | 4.7% | +0.0039 (0.0071) | +0.0014 (0.0040) | −0.0016 (0.0037) | −0.0107 (0.0060) | +0.0001 (0.0023) |
| M | 51-100 | 21.8% | 21.4% | 0.3480 | 23.1% | +0.0001 (0.0036) | +0.0035 (0.0023) | +0.0008 (0.0020) | −0.0181 (0.0029) | −0.0007 (0.0011) |
| M | 201-300 | 16.1% | 17.9% | 0.2629 | 19.3% | −0.0012 (0.0032) | +0.0021 (0.0016) | +0.0049 (0.0017) | −0.0136 (0.0025) | +0.0028 (0.0010) |
| H | 0 | 1.3% | 0.9% | 0.3690 | 1.4% | +0.0064 (0.0167) | +0.0091 (0.0082) | +0.0086 (0.0064) | +0.0013 (0.0134) | +0.0052 (0.0050) |
| H | 1-19 | 3.9% | 3.7% | 0.4079 | 3.7% | **+0.0308 (0.0099)** | +0.0042 (0.0049) | +0.0069 (0.0044) | −0.0130 (0.0077) | +0.0042 (0.0030) |
| H | 64-191 | 36.9% | 37.8% | 0.3933 | 36.4% | +0.0078 (0.0029) | +0.0022 (0.0017) | +0.0004 (0.0016) | −0.0173 (0.0023) | −0.0002 (0.0009) |
| H | 256 (cap) | 28.7% | 28.8% | 0.3736 | 29.3% | **−0.0145 (0.0030)** | +0.0047 (0.0018) | +0.0042 (0.0017) | −0.0166 (0.0025) | +0.0010 (0.0010) |
| user | uid0 | 4.6% | 4.1% | 0.3951 | 4.5% | +0.0258 (0.0089) | +0.0051 (0.0045) | +0.0019 (0.0043) | −0.0183 (0.0065) | +0.0031 (0.0023) |
| user | 0 train feeds | 65.2% | 66.8% | 0.3880 | 65.0% | +0.0010 (0.0021) | +0.0024 (0.0012) | +0.0028 (0.0012) | −0.0158 (0.0017) | +0.0010 (0.0007) |
| user | 1 train feed | 19.3% | 18.8% | 0.3827 | 19.4% | −0.0011 (0.0039) | +0.0066 (0.0023) | +0.0023 (0.0021) | −0.0195 (0.0032) | −0.0014 (0.0012) |
| user | 2-4 train feeds | 10.0% | 9.5% | 0.3777 | 10.1% | −0.0138 (0.0053) | +0.0033 (0.0029) | +0.0041 (0.0029) | −0.0151 (0.0043) | +0.0006 (0.0020) |
| cold share | < 15% | 9.7% | 1.4% | 0.4452 | 8.8% | −0.0068 (0.0059) | +0.0052 (0.0036) | +0.0038 (0.0032) | −0.0163 (0.0050) | +0.0007 (0.0019) |
| cold share | 30-45% | 34.5% | 51.4% | 0.3717 | 35.3% | −0.0044 (0.0028) | +0.0021 (0.0016) | +0.0025 (0.0016) | −0.0124 (0.0023) | +0.0006 (0.0009) |
| cold share | ≥ 45% | 8.3% | 29.1% | 0.4635 | 7.3% | **+0.0255 (0.0068)** | +0.0021 (0.0034) | +0.0045 (0.0034) | −0.0228 (0.0052) | +0.0016 (0.0020) |
| hour (MSK) | 18-23 | 25.1% | 31.2% | 0.3712 | 25.7% | −0.0045 (0.0033) | +0.0043 (0.0019) | +0.0019 (0.0019) | −0.0146 (0.0028) | +0.0004 (0.0010) |
| day | Tue | 52.3% | | 0.3858 | 52.3% | | +0.0047 (0.0014) | +0.0028 (0.0013) | −0.0168 (0.0019) | −0.0002 (0.0007) |
| day | Thu | 47.7% | | 0.3865 | 47.7% | | +0.0021 (0.0014) | +0.0027 (0.0013) | −0.0164 (0.0020) | +0.0014 (0.0008) |
| NN/CB rank ρ | ≤ 0.7 | 13.1% | | 0.4233 | 12.3% | −0.0152 (0.0049) | +0.0040 (0.0027) | +0.0074 (0.0025) | −0.0264 (0.0047) | +0.0031 (0.0015) |
| NN/CB rank ρ | > 0.9 | 15.3% | | 0.4560 | 13.6% | +0.0309 (0.0048) | +0.0056 (0.0027) | +0.0018 (0.0027) | −0.0134 (0.0034) | −0.0020 (0.0015) |

Readings:
- **Tue vs Thu.** Identical NDCG: 0.3858 vs 0.3865. The fine-tune gain differs (+0.0047 vs +0.0021), but the gap is only 1.3 SE.
- **Hour.** Evening and night feeds score 0.371-0.373 against 0.39-0.40 for daytime. The gap disappears once M is controlled for. Test has more evening feeds (31.2% vs 25.1%).
- **H = 256** is the only large observable segment that is significantly bad after adjustment. The effect comes from H256 users with 0 train feeds (−0.0175 (0.0041), n = 1,993) and with 2-4 train feeds (−0.0178 (0.0067)). Users with exactly 1 train feed are not significantly affected (−0.0048 (0.0060)). Earlier-snapshot history did not help them (§6).
- **Short histories and user 0 do better than M predicts:** H 1-19 +0.031, uid0 +0.026.
- **Model disagreement.** Low NN/CB agreement (ρ ≤ 0.7) is where the stacker gains most (+0.0074) and the blend helps (+0.0031). High agreement (ρ > 0.9) is where the blend hurts (−0.0020). Routing by ρ in CV still gives −0.0005 (0.0003).

## 4. Diagnostic segments (need labels; they show the type of error)

| segment | bucket | share | NN oof | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof − tr | stk − tr | CB − oof | blend − oof |
|---|---|---|---|---|---|---|---|---|---|---|---|
| positive's f3 category rank | #1 | 52.5% | 0.4311 | 48.6% | +0.0421 (0.0025) | 0.050 | 0.387 | +0.0038 (0.0015) | +0.0039 (0.0014) | −0.0056 (0.0021) | +0.0017 (0.0008) |
| | #2 | 17.7% | 0.3923 | 17.5% | −0.0035 (0.0040) | 0.185 | 0.128 | −0.0004 (0.0023) | +0.0024 (0.0023) | −0.0180 (0.0034) | +0.0017 (0.0013) |
| | #3 | 9.6% | 0.3588 | 10.1% | −0.0380 (0.0052) | 0.267 | 0.052 | −0.0011 (0.0029) | +0.0003 (0.0025) | −0.0363 (0.0041) | +0.0013 (0.0016) |
| | #4-6 | 13.4% | 0.3020 | 15.2% | −0.0802 (0.0035) | 0.383 | 0.010 | +0.0085 (0.0020) | +0.0014 (0.0018) | −0.0368 (0.0030) | −0.0037 (0.0011) |
| | #7+ | 6.8% | 0.2274 | 8.6% | −0.1037 (0.0032) | 0.520 | −0.017 | +0.0073 (0.0020) | +0.0010 (0.0018) | −0.0300 (0.0027) | −0.0035 (0.0011) |
| positive's f1 percentile | top | 12.3% | 0.5563 | 8.9% | +0.1011 (0.0057) | 0.082 | 0.253 | −0.0012 (0.0033) | +0.0047 (0.0032) | +0.0026 (0.0049) | +0.0045 (0.0019) |
| | > 60% | 17.4% | 0.2817 | 20.3% | −0.1021 (0.0027) | 0.414 | 0.028 | +0.0113 (0.0017) | +0.0032 (0.0014) | −0.0404 (0.0023) | −0.0041 (0.0009) |
| positive category in history | no | 6.5% | 0.2846 | 7.6% | −0.0872 (0.0055) | 0.427 | 0.060 | +0.0075 (0.0024) | −0.0009 (0.0018) | −0.0176 (0.0040) | −0.0031 (0.0014) |
| positive category's share of history | ≤ 5% (in history) | 12.0% | 0.3121 | 13.5% | −0.0669 (0.0041) | 0.402 | 0.012 | +0.0053 (0.0021) | −0.0011 (0.0018) | −0.0252 (0.0032) | −0.0027 (0.0013) |
| | > 50% | 29.6% | 0.4203 | 27.9% | +0.0287 (0.0031) | 0.033 | 0.427 | +0.0071 (0.0020) | +0.0051 (0.0019) | −0.0102 (0.0026) | +0.0009 (0.0011) |
| positive category's share of feed | ≤ 5% | 6.2% | 0.2398 | 7.7% | −0.0803 (0.0047) | 0.579 | −0.025 | +0.0052 (0.0022) | +0.0011 (0.0022) | −0.0276 (0.0034) | −0.0032 (0.0011) |
| positives' coldness | all cold | 17.4% | 0.3575 | 18.2% | −0.0202 (0.0042) | 0.166 | 0.276 | **−0.0159 (0.0020)** | **−0.0217 (0.0020)** | −0.0432 (0.0031) | −0.0017 (0.0011) |
| | warm | 71.0% | 0.3895 | 70.6% | +0.0044 (0.0021) | 0.188 | 0.215 | +0.0088 (0.0013) | +0.0101 (0.0012) | −0.0088 (0.0017) | +0.0015 (0.0007) |
| npos | 1 | 74.5% | 0.3816 | 75.1% | +0.0000 (0.0021) | 0.198 | 0.224 | +0.0029 (0.0012) | +0.0027 (0.0012) | −0.0173 (0.0017) | +0.0008 (0.0007) |
| | 2 | 15.4% | 0.3811 | 15.5% | +0.0000 (0.0035) | 0.113 | 0.247 | +0.0049 (0.0020) | +0.0037 (0.0019) | −0.0138 (0.0029) | +0.0000 (0.0012) |
| | 6+ | 1.8% | 0.5195 | 1.4% | +0.1128 (0.0083) | 0.032 | 0.233 | +0.0057 (0.0041) | +0.0028 (0.0038) | −0.0164 (0.0052) | −0.0021 (0.0019) |

**Positive category** (single-category feeds; "multi" = 16.3%):
- Loss tracks feed share for every category (lift 0.88-1.06).
- The fine-tune gain changes sign by category: cat3 −0.0070 (0.0038), cat15 −0.0059 (0.0058), cat13 +0.0068 (0.0030), cat2 +0.0066 (0.0036). This is not a static category prior: adding a category one-hot to NN tr in CV *hurts*, −0.0009 (0.0004).
- CB beats NN oof only in cat15 (+0.0083 (0.0074)) and cat2 (+0.0032 (0.0049)), both n.s. It collapses in cat50 (−0.0429 (0.0057)).

**Blend vs NN oof.** The stacker pulls toward f-features. That helps when the positive is f1-top (+0.0045) and hurts when it is off-f3 or f1 > 60% (−0.004).

## 5. Pairwise structure (where the models actually fail)

Pairwise AUC, pair-weighted, all pairs within a feed (`pair_auc.log`):

| score | warm+ vs warm− | warm+ vs cold− | cold+ vs warm− | cold+ vs cold− |
|---|---|---|---|---|
| NN oof | 0.6983 | 0.8024 | **0.6594** | 0.7570 |
| NN tr | 0.6903 | 0.7921 | 0.6636 | 0.7547 |
| stk_clean | 0.6957 | 0.8075 | 0.6479 | 0.7557 |
| CB | 0.6704 | 0.8165 | 0.5752 | 0.7337 |
| maxsim32 | 0.6387 | 0.6985 | 0.6464 | 0.6999 |
| f4 (= listing age) | 0.5065 | 0.8421 | 0.2169 | 0.5911 |
| pair share | 54.7% | 21.8% | 16.1% | 7.3% |

Same-category pairs only:

| score | warm+ vs warm− | warm+ vs cold− | cold+ vs warm− | cold+ vs cold− |
|---|---|---|---|---|
| NN oof | 0.657 | 0.763 | **0.599** | 0.702 |
| NN tr | 0.644 | 0.751 | 0.601 | 0.697 |
| stk_clean | 0.653 | 0.779 | 0.574 | 0.700 |
| CB | 0.630 | 0.824 | 0.444 | 0.677 |

Cross-category pairs: NN oof 0.713 / 0.816 / 0.683 / 0.784.

Readings:
- The val-label components improve warm+ pairs (+0.008 to +0.016) and degrade cold+ vs warm− pairs (−0.004 to −0.016).
- On cold+ vs warm− pairs, item-age (f4) and popularity (f5) are strongly *anti*-informative. In "item has a history" terms, warmth is the right prior on average (cold positive rate 1.24% vs warm 1.72%), but it costs ranks whenever a fresh listing is the click.

Within-(feed, category) AUC on groups with all-cold positives (`within_cat_auc.log`):
- NN oof 0.625, NN tr 0.624, stk_clean 0.608, CB 0.520.
- Content similarity is the only signal that is equally good on cold items: cos_h10 0.604, maxsim32 0.607.
- Coarse cluster CTR is useless: `clu_ic1024_ctr` scores 0.507 overall and 0.507 on cold groups.

## 6. Specialized-treatment tests on existing signals (all negative)

All tests use nested CV: 5-fold GroupKFold by user, with the weight or choice made on the train folds only.

| test | result vs reference |
|---|---|
| Per-bucket blend weight, NN oof vs stk (grid 0-1), routed by M / H / user / cold share / n_cat / hist-top-cat share / cand-cat-in-hist / ρ / hour / day | −0.0010 / −0.0008 / −0.0004 / −0.0006 / −0.0005 / −0.0009 / −0.0010 / −0.0005 / −0.0008 / −0.0000 vs global weight (SE 0.0003-0.0004) |
| Same with NN oof vs CB (w ≤ 0.3) and NN oof vs NN tr | all between −0.0012 and +0.0000 |
| Cold offset z + δ·cold (δ ∈ {0, .1, .2, .3}) | NN oof −0.0002 (0.0001); stk_clean +0.0004 (0.0003); blend 0.0000 |
| What the fine-tune learned: LR on z(NN tr) + category one-hot / + warmness (in train cand, in train hist, log counts) / both | −0.0009 (0.0004) / +0.0009 (0.0006) / +0.0003 (0.0008); NN oof itself gains +0.0034 |
| Intent regime ("positive not in f3-top category") predicted from observable feed features: f3 gap, f3-top category's candidate share, NN category margin, n_cat, M, H, history concentration, uid0, ρ | CV AUC 0.772 (f3-top candidate share alone 0.751). NDCG by predicted tertile is flat (0.393 / 0.373 / 0.393) |
| Regime-routed weights, by predicted tertile: z(f3) boost / CB weight / stk weight | +0.0000 / −0.0000 (0.0002) / +0.0001 (0.0003) vs global |
| Category-intent LR (category level; NN max/mean, f3, f1max, candidate share, history counts all / last 1 / 5 / 20 / decayed / event type 1) as boost z + w·logit P(cat) | category top-1 0.513 → 0.570, but off-f3 top-1 stays 0.210. Boost w = 0.1: +0.0001 (0.0003); f3-top feeds +0.0055, off-f3 feeds −0.0060. Adding history signals to NN + f changes nothing (0.570 → 0.570) |
| Same user's earlier train snapshots (allowed): about 107 old-only items for 28.6% of feeds; max-sim, mean-cos and category share to them | NN oof + w·z: −0.0001 / −0.0003 / −0.0002 (SE 0.0002-0.0003); on train-seen feeds −0.0003 / −0.0010 / −0.0009. The positive category was absent from the val history but present in old history in only 58 of 4,315 feeds |

Category-ranking quality by signal (feeds with ≥ 2 categories; "top-1" = the best-scoring category holds a positive):

| signal | top-1 overall | top-1, f3-top regime | top-1, off-f3 regime | MRR, off-f3 regime |
|---|---|---|---|---|
| f3 | 0.519 | 0.990 | 0.000 | 0.329 |
| NN oof max | 0.513 | 0.789 | 0.210 | 0.454 |
| CB max | 0.530 | 0.855 | 0.173 | 0.433 |
| candidate share of category | 0.523 | 0.821 | 0.195 | 0.437 |
| history, decayed (hl 10) | 0.504 | 0.832 | 0.143 | 0.401 |
| history, newest event | 0.452 | 0.714 | 0.163 | 0.249 |

## 7. The three segments with the largest specialized-treatment potential

Ranked by total headroom × evidence that information is missing. Reweighting existing models is ruled out by §6, so every proposal brings new information.

### S1. Dominant-intent feeds: within-category ranking (52.5% of feeds, 48.6% of loss, 0.203 of the 0.231 within-orc headroom)

- **Evidence.**
  - The cat-orc gap is only 0.050 per feed, but the within-orc gap is 0.387 per feed.
  - NN same-category AUC on warm-warm pairs is 0.657, against 0.713 on cross-category pairs.
  - In the warm-positive sub-cell (34.9% of feeds), train-derived item statistics carry signal the NN underuses: CB − NN oof = +0.0076 (0.0028) and stk − NN tr = +0.0142 (0.0019).
  - The cell cannot be targeted by routing (§6), so the information has to reach the model as features for all feeds.
  - No item-level co-visitation feature exists in data/feats. The only related family, `clu_pmi*`, is cluster-level and coarse.
- **Test T1a: item-to-item co-visitation.**
  - Source: train histories, plus other users' same-split histories, past-only and normalised per 10K feeds.
  - Feature: for candidate j, Σ_{i ∈ history} 0.5^(hpos_i/10) · covis(i, j). Add a same-category-only variant, and the max over i.
  - Gates: (a) within-(feed, category) AUC on f3-top-regime val feeds ≥ 0.60 single, with a partial gain over z(NN oof); (b) z(NN oof) + w·z(feature) in nested CV ≥ +0.001 and > 2 SE; (c) an equal or larger gain in the residual stacker.
  - Expected: +0.001 to +0.003.
- **Test T1b: within-category re-ranker.**
  - Model: LambdaRank with group = (feed, category), trained on train-feed groups with ≥ 1 positive.
  - Features: item-level only. Age (f4), f5, f1, sims to same-category history events, it_hist_*_luo, plus covis from T1a. Exclude f3 and other category-level features.
  - Use: as a residual, z(NN) + w·z(within).
  - Report: per regime (f3-top vs off-f3) and per positive coldness, using the §2 table as the yardstick.

### S2. Fresh-listing positives (17.4% of val feeds, 22.1% at the test mix, NN oof 0.3575 vs 0.3895 on warm-positive feeds)

- **Evidence.**
  - Every val-label component loses here: NN oof −0.0159 (0.0020), stk −0.0217, stk_clean −0.0224 vs NN tr.
  - Cold+ vs warm− pairs are the worst pair type for every model.
  - A bias correction does not fix it (§6).
  - Content similarity is the only signal that holds up on cold items. Coarse cluster CTR is useless (AUC 0.51).
  - The segment grows on test, and private has more cold items than public.
- **Test T2a: fine-grained content-kNN quality prior.**
  - Feature: for every candidate, warm or cold, the position-debiased positive rate (or COEC) of its k = 20-50 nearest train candidates in the same category, in item-embedding space, excluding the item itself. Warm and cold items then get the same definition.
  - Compute: faiss IVF on 39M train items, on GPU N or on CPU.
  - Gates: same-category cold+ vs warm− AUC > 0.62 (NN oof 0.599); residual gain on all-cold-positive feeds with no loss on warm-positive feeds; total gain > 2 SE.
- **Test T2b: ID-dropout in the val-label stage.**
  - Method: in the fine-tune and the stacker, null the ID-based inputs (ALS, it_hist_*, it_tr_*, item-ID embedding) for a random 35-40% of warm candidates, matching the test cold rate.
  - Selection: select on the **test-mix-weighted val** (reweight feeds by M × cold-share bucket; code in `cv_tests.py` §2), not on raw val. On raw val, expect the gain to look smaller than it would be on test.
- **Expected:** +0.001 to +0.002 on test. Part of it is robustness that raw val under-states: the fine-tune gain already falls from +0.0034 to +0.0026 under the test mix.

### S3. Intent-switch feeds: cross-category intent (47.5% of feeds, 51.4% of loss, 0.145 of the 0.171 cat-orc headroom)

- **Evidence.**
  - Largest cross-category headroom, but none of the current inputs reaches it.
  - Off-f3 category top-1 is ≤ 0.21 for every signal and for a CV LR on all of them.
  - History recency adds nothing beyond NN + f3.
  - Earlier snapshots recover the positive category in only 58 of 4,315 feeds.
  - The regime is predictable (AUC 0.77), but knowing it does not change the best weights.
  - The earlier NN aux category head was neutral, which is consistent with this.
- **Test T3, the only remaining new-information idea: cross-user category-switch prior.**
  - Data: train feeds (3.1M, positives only).
  - Target: P(clicked category = c | f3-top category, f3 rank of c, last-k history categories, hour, day-of-week, c's candidate share).
  - Model: a 52-way LightGBM or a factorised table.
  - **Hard gate before any integration:** off-f3 category MRR on val > 0.47 (NN oof max: 0.454) **and** off-f3 top-1 > 0.24. If either fails, stop. If both pass, use log P(c) as a stacker feature and report per regime.
  - Expected: 0 to +0.002, with low probability of success.

### Lower priority, noted for completeness
- **H = 256 feeds (28.7%), M-adjusted −0.0145 (0.0030).** The obvious cure, longer history from earlier snapshots, failed (§6). The users most affected have no train presence (H256 with 0 train feeds: −0.0175). Nothing concrete to propose beyond what S1 and S2 would bring.
- **M > 200 feeds.** The blend gains +0.0028 (0.0010) here and the stacker +0.0049 (0.0017) over NN tr, but routing by M is −0.0010 in CV.

## 8. Files

- `build_feed_table.py`: per-feed table (`feed_table.parquet`, `test_segments.parquet`): segment variables, per-model NDCG, oracles, leak flag.
- `segdefs.py`, `segment_tables.py`, `print_md.py`: per-segment tables (`segments.csv`, `report_tables.md`, `segment_tables.log`).
- `cv_tests.py`: M-adjusted deviations (`m_adjusted.csv`), test-mix reweighting, LR decomposition of the fine-tune gain, segment-routed blends (`cv_tests.log`).
- `error_map.py`: §2 table.
- `cat_intent.py`: category-signal quality and the category-boost test.
- `regime_cold_tests.py`: cold offset, regime predictability, regime-routed weights.
- `pair_auc.py`, `within_cat_auc.py`: pairwise and within-category AUC, H256 × user, test-mix cold share.
- `old_history.py`, `old_history_sim.py`: earlier-snapshot history coverage and value test.

## Appendix: all segment tables

Columns:
- **share test**: share of test feeds (observable segments only).
- **loss share**: the bucket's share of total NN oof loss.
- **M-adj dev**: NN oof NDCG minus the mean of its M-ventile × npos cell.
- **cat-orc gap / within-orc gap**: per-feed gains of the two oracles.

**M**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M5-10 | 8.4% | 7.6% | 0.6593 | 0.6580 | 4.7% | +0.0039 (0.0071) | 0.193 | 0.110 | +0.0014 (0.0040) | -0.0016 (0.0037) | -0.0107 (0.0060) | +0.0001 (0.0023) |
| M11-25 | 15.0% | 13.8% | 0.5061 | 0.5010 | 12.1% | -0.0018 (0.0052) | 0.223 | 0.160 | +0.0050 (0.0030) | +0.0029 (0.0028) | -0.0145 (0.0043) | +0.0008 (0.0018) |
| M26-50 | 18.1% | 18.3% | 0.4168 | 0.4145 | 17.2% | -0.0003 (0.0043) | 0.211 | 0.205 | +0.0023 (0.0025) | +0.0036 (0.0025) | -0.0198 (0.0035) | +0.0011 (0.0013) |
| M51-100 | 21.8% | 21.4% | 0.3480 | 0.3445 | 23.1% | +0.0001 (0.0036) | 0.171 | 0.250 | +0.0035 (0.0023) | +0.0008 (0.0020) | -0.0181 (0.0029) | -0.0007 (0.0011) |
| M101-200 | 20.6% | 21.1% | 0.2967 | 0.2916 | 23.6% | +0.0009 (0.0033) | 0.142 | 0.274 | +0.0051 (0.0018) | +0.0040 (0.0017) | -0.0184 (0.0026) | -0.0001 (0.0009) |
| M201-300 | 16.1% | 17.9% | 0.2629 | 0.2609 | 19.3% | -0.0012 (0.0032) | 0.106 | 0.308 | +0.0021 (0.0016) | +0.0049 (0.0017) | -0.0136 (0.0025) | +0.0028 (0.0010) |

**H**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H0 | 1.3% | 0.9% | 0.3690 | 0.3599 | 1.4% | +0.0064 (0.0167) | 0.156 | 0.270 | +0.0091 (0.0082) | +0.0086 (0.0064) | +0.0013 (0.0134) | +0.0052 (0.0050) |
| H1-19 | 3.9% | 3.7% | 0.4079 | 0.4037 | 3.7% | +0.0308 (0.0099) | 0.122 | 0.319 | +0.0042 (0.0049) | +0.0069 (0.0044) | -0.0130 (0.0077) | +0.0042 (0.0030) |
| H20-63 | 8.4% | 8.4% | 0.3937 | 0.3950 | 8.3% | +0.0108 (0.0061) | 0.152 | 0.262 | -0.0013 (0.0035) | -0.0014 (0.0032) | -0.0121 (0.0047) | -0.0002 (0.0016) |
| H64-191 | 36.9% | 37.8% | 0.3933 | 0.3911 | 36.4% | +0.0078 (0.0029) | 0.171 | 0.222 | +0.0022 (0.0017) | +0.0004 (0.0016) | -0.0173 (0.0023) | -0.0002 (0.0009) |
| H192-255 | 20.8% | 20.5% | 0.3846 | 0.3794 | 20.8% | -0.0042 (0.0037) | 0.174 | 0.218 | +0.0052 (0.0022) | +0.0056 (0.0021) | -0.0189 (0.0030) | +0.0008 (0.0012) |
| H256 | 28.7% | 28.8% | 0.3736 | 0.3690 | 29.3% | -0.0145 (0.0030) | 0.184 | 0.229 | +0.0047 (0.0018) | +0.0042 (0.0017) | -0.0166 (0.0025) | +0.0010 (0.0010) |

**user**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uid0 | 4.6% | 4.1% | 0.3951 | 0.3900 | 4.5% | +0.0258 (0.0089) | 0.116 | 0.306 | +0.0051 (0.0045) | +0.0019 (0.0043) | -0.0183 (0.0065) | +0.0031 (0.0023) |
| tr0 | 65.2% | 66.8% | 0.3880 | 0.3856 | 65.0% | +0.0010 (0.0021) | 0.175 | 0.225 | +0.0024 (0.0012) | +0.0028 (0.0012) | -0.0158 (0.0017) | +0.0010 (0.0007) |
| tr1 | 19.3% | 18.8% | 0.3827 | 0.3761 | 19.4% | -0.0011 (0.0039) | 0.171 | 0.230 | +0.0066 (0.0023) | +0.0023 (0.0021) | -0.0195 (0.0032) | -0.0014 (0.0012) |
| tr2-4 | 10.0% | 9.5% | 0.3777 | 0.3744 | 10.1% | -0.0138 (0.0053) | 0.176 | 0.232 | +0.0033 (0.0029) | +0.0041 (0.0029) | -0.0151 (0.0043) | +0.0006 (0.0020) |
| tr5+ | 0.9% | 0.9% | 0.3715 | 0.3664 | 0.9% | -0.0278 (0.0162) | 0.138 | 0.307 | +0.0051 (0.0086) | -0.0009 (0.0102) | -0.0184 (0.0122) | -0.0016 (0.0050) |

**cold_share**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold<15% | 9.7% | 1.4% | 0.4452 | 0.4400 | 8.8% | -0.0068 (0.0059) | 0.210 | 0.178 | +0.0052 (0.0036) | +0.0038 (0.0032) | -0.0163 (0.0050) | +0.0007 (0.0019) |
| cold15-30% | 47.5% | 18.1% | 0.3710 | 0.3667 | 48.7% | +0.0001 (0.0024) | 0.180 | 0.218 | +0.0043 (0.0014) | +0.0025 (0.0013) | -0.0185 (0.0020) | +0.0004 (0.0008) |
| cold30-45% | 34.5% | 51.4% | 0.3717 | 0.3696 | 35.3% | -0.0044 (0.0028) | 0.156 | 0.256 | +0.0021 (0.0016) | +0.0025 (0.0016) | -0.0124 (0.0023) | +0.0006 (0.0009) |
| cold>=45% | 8.3% | 29.1% | 0.4635 | 0.4614 | 7.3% | +0.0255 (0.0068) | 0.142 | 0.260 | +0.0021 (0.0034) | +0.0045 (0.0034) | -0.0228 (0.0052) | +0.0016 (0.0020) |

**hour**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h00-05 | 9.2% | 10.0% | 0.3726 | 0.3699 | 9.4% | -0.0016 (0.0054) | 0.163 | 0.232 | +0.0027 (0.0031) | +0.0051 (0.0029) | -0.0177 (0.0043) | +0.0016 (0.0018) |
| h06-09 | 19.0% | 18.0% | 0.3903 | 0.3873 | 18.9% | +0.0034 (0.0040) | 0.172 | 0.232 | +0.0030 (0.0024) | +0.0024 (0.0023) | -0.0162 (0.0033) | +0.0006 (0.0013) |
| h10-13 | 23.5% | 19.5% | 0.3983 | 0.3935 | 23.1% | +0.0020 (0.0036) | 0.174 | 0.219 | +0.0047 (0.0020) | +0.0041 (0.0018) | -0.0162 (0.0029) | +0.0003 (0.0010) |
| h14-17 | 23.2% | 21.3% | 0.3919 | 0.3901 | 23.0% | +0.0007 (0.0035) | 0.172 | 0.238 | +0.0018 (0.0021) | +0.0017 (0.0020) | -0.0190 (0.0029) | +0.0008 (0.0012) |
| h18-23 | 25.1% | 31.2% | 0.3712 | 0.3669 | 25.7% | -0.0045 (0.0033) | 0.171 | 0.234 | +0.0043 (0.0019) | +0.0019 (0.0019) | -0.0146 (0.0028) | +0.0004 (0.0010) |

**day**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tue | 52.3% | - | 0.3858 | 0.3811 | 52.3% | -0.0005 (0.0023) | 0.172 | 0.230 | +0.0047 (0.0014) | +0.0028 (0.0013) | -0.0168 (0.0019) | -0.0002 (0.0007) |
| Thu | 47.7% | - | 0.3865 | 0.3844 | 47.7% | +0.0005 (0.0025) | 0.171 | 0.231 | +0.0021 (0.0014) | +0.0027 (0.0013) | -0.0164 (0.0020) | +0.0014 (0.0008) |

**n_cat**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ncat1-4 | 7.6% | 6.8% | 0.5999 | 0.5978 | 4.9% | +0.0052 (0.0073) | 0.114 | 0.237 | +0.0021 (0.0043) | -0.0004 (0.0041) | -0.0124 (0.0062) | +0.0000 (0.0024) |
| ncat5-8 | 22.6% | 21.0% | 0.4743 | 0.4710 | 19.3% | -0.0023 (0.0040) | 0.181 | 0.220 | +0.0033 (0.0024) | +0.0006 (0.0022) | -0.0152 (0.0033) | +0.0000 (0.0013) |
| ncat9-12 | 26.9% | 26.2% | 0.3786 | 0.3734 | 27.2% | +0.0033 (0.0033) | 0.179 | 0.241 | +0.0052 (0.0019) | +0.0048 (0.0018) | -0.0176 (0.0028) | +0.0008 (0.0010) |
| ncat13-18 | 29.4% | 30.3% | 0.3145 | 0.3115 | 32.8% | -0.0045 (0.0028) | 0.175 | 0.236 | +0.0030 (0.0017) | +0.0036 (0.0015) | -0.0176 (0.0022) | +0.0010 (0.0009) |
| ncat19+ | 13.6% | 15.7% | 0.2904 | 0.2885 | 15.7% | +0.0041 (0.0041) | 0.167 | 0.215 | +0.0019 (0.0020) | +0.0022 (0.0020) | -0.0170 (0.0031) | +0.0006 (0.0011) |

**hist_top_cat_share**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| htop=0(H0) | 1.3% | 0.9% | 0.3690 | 0.3599 | 1.4% | +0.0064 (0.0167) | 0.156 | 0.270 | +0.0091 (0.0082) | +0.0086 (0.0064) | +0.0013 (0.0134) | +0.0052 (0.0050) |
| htop<=.33 | 24.0% | 24.4% | 0.3816 | 0.3823 | 24.1% | -0.0077 (0.0035) | 0.231 | 0.125 | -0.0007 (0.0020) | -0.0005 (0.0018) | -0.0152 (0.0027) | +0.0003 (0.0011) |
| htop.33-.5 | 33.4% | 33.0% | 0.3850 | 0.3821 | 33.5% | +0.0013 (0.0030) | 0.186 | 0.189 | +0.0029 (0.0017) | +0.0027 (0.0016) | -0.0190 (0.0024) | +0.0002 (0.0009) |
| htop.5-.7 | 22.4% | 23.5% | 0.3874 | 0.3817 | 22.3% | +0.0019 (0.0036) | 0.151 | 0.263 | +0.0057 (0.0021) | +0.0046 (0.0021) | -0.0193 (0.0030) | +0.0008 (0.0011) |
| htop>.7 | 18.9% | 18.3% | 0.3935 | 0.3871 | 18.7% | +0.0049 (0.0039) | 0.096 | 0.398 | +0.0064 (0.0024) | +0.0043 (0.0022) | -0.0121 (0.0033) | +0.0011 (0.0014) |

**cand_cat_in_hist**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cih<=.5 | 3.4% | 3.0% | 0.4209 | 0.4219 | 3.2% | +0.0171 (0.0105) | 0.170 | 0.249 | -0.0011 (0.0050) | +0.0028 (0.0047) | +0.0021 (0.0085) | +0.0045 (0.0033) |
| cih.5-.8 | 22.2% | 22.2% | 0.4069 | 0.4038 | 21.5% | +0.0162 (0.0038) | 0.173 | 0.237 | +0.0031 (0.0021) | +0.0009 (0.0020) | -0.0198 (0.0030) | +0.0000 (0.0012) |
| cih.8-.95 | 53.4% | 55.0% | 0.3601 | 0.3564 | 55.7% | -0.0049 (0.0022) | 0.172 | 0.230 | +0.0037 (0.0013) | +0.0034 (0.0013) | -0.0173 (0.0018) | +0.0007 (0.0007) |
| cih>.95 | 20.9% | 19.8% | 0.4247 | 0.4210 | 19.6% | -0.0075 (0.0038) | 0.170 | 0.223 | +0.0037 (0.0023) | +0.0031 (0.0021) | -0.0143 (0.0032) | +0.0002 (0.0011) |

**rho_nn_cb**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rho<=.7 | 13.1% | - | 0.4233 | 0.4193 | 12.3% | -0.0152 (0.0049) | 0.229 | 0.127 | +0.0040 (0.0027) | +0.0074 (0.0025) | -0.0264 (0.0047) | +0.0031 (0.0015) |
| rho.7-.8 | 25.9% | - | 0.3618 | 0.3592 | 27.0% | -0.0075 (0.0033) | 0.197 | 0.181 | +0.0026 (0.0019) | +0.0024 (0.0018) | -0.0182 (0.0027) | +0.0015 (0.0010) |
| rho.8-.9 | 45.7% | - | 0.3658 | 0.3629 | 47.2% | -0.0018 (0.0024) | 0.156 | 0.264 | +0.0030 (0.0015) | +0.0020 (0.0014) | -0.0139 (0.0020) | +0.0002 (0.0008) |
| rho>.9 | 15.3% | - | 0.4560 | 0.4503 | 13.6% | +0.0309 (0.0048) | 0.126 | 0.303 | +0.0056 (0.0027) | +0.0018 (0.0027) | -0.0134 (0.0034) | -0.0020 (0.0015) |

**npos**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| npos1 | 74.5% | - | 0.3816 | 0.3787 | 75.1% | +0.0000 (0.0021) | 0.198 | 0.224 | +0.0029 (0.0012) | +0.0027 (0.0012) | -0.0173 (0.0017) | +0.0008 (0.0007) |
| npos2 | 15.4% | - | 0.3811 | 0.3762 | 15.5% | +0.0000 (0.0035) | 0.113 | 0.247 | +0.0049 (0.0020) | +0.0037 (0.0019) | -0.0138 (0.0029) | +0.0000 (0.0012) |
| npos3 | 5.1% | - | 0.3943 | 0.3920 | 5.0% | -0.0446 (0.0052) | 0.079 | 0.260 | +0.0023 (0.0030) | -0.0016 (0.0029) | -0.0128 (0.0043) | -0.0007 (0.0015) |
| npos4-5 | 3.2% | - | 0.4256 | 0.4178 | 3.0% | +0.0060 (0.0061) | 0.059 | 0.269 | +0.0078 (0.0033) | +0.0073 (0.0034) | -0.0197 (0.0046) | +0.0010 (0.0017) |
| npos6+ | 1.8% | - | 0.5195 | 0.5138 | 1.4% | +0.1128 (0.0083) | 0.032 | 0.233 | +0.0057 (0.0041) | +0.0028 (0.0038) | -0.0164 (0.0052) | -0.0021 (0.0019) |

**pos_cat**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cat13 | 12.4% | - | 0.3810 | 0.3742 | 12.5% | -0.0017 (0.0047) | 0.079 | 0.402 | +0.0068 (0.0030) | +0.0030 (0.0032) | -0.0220 (0.0044) | +0.0006 (0.0016) |
| cat2 | 9.6% | - | 0.4016 | 0.3950 | 9.4% | +0.0333 (0.0057) | 0.094 | 0.364 | +0.0066 (0.0036) | +0.0029 (0.0034) | +0.0032 (0.0049) | -0.0004 (0.0020) |
| cat3 | 7.6% | - | 0.3998 | 0.4068 | 7.4% | +0.0285 (0.0066) | 0.128 | 0.306 | -0.0070 (0.0038) | -0.0082 (0.0038) | -0.0242 (0.0053) | +0.0003 (0.0020) |
| cat50 | 6.0% | - | 0.3780 | 0.3782 | 6.1% | -0.0133 (0.0075) | 0.256 | 0.106 | -0.0003 (0.0042) | +0.0014 (0.0037) | -0.0429 (0.0057) | +0.0001 (0.0024) |
| cat6 | 4.7% | - | 0.3666 | 0.3663 | 4.8% | -0.0073 (0.0081) | 0.213 | 0.159 | +0.0003 (0.0049) | -0.0007 (0.0040) | -0.0222 (0.0070) | +0.0011 (0.0025) |
| cat7 | 3.9% | - | 0.3663 | 0.3670 | 4.0% | -0.0126 (0.0086) | 0.221 | 0.171 | -0.0007 (0.0055) | +0.0017 (0.0049) | -0.0069 (0.0074) | +0.0019 (0.0024) |
| cat10 | 3.6% | - | 0.4604 | 0.4533 | 3.2% | +0.0517 (0.0113) | 0.189 | 0.208 | +0.0071 (0.0060) | +0.0071 (0.0051) | -0.0158 (0.0079) | +0.0013 (0.0033) |
| cat15 | 3.1% | - | 0.3513 | 0.3572 | 3.3% | -0.0352 (0.0077) | 0.118 | 0.345 | -0.0059 (0.0058) | +0.0016 (0.0053) | +0.0083 (0.0074) | +0.0037 (0.0028) |
| cat11 | 2.5% | - | 0.4118 | 0.4152 | 2.4% | +0.0259 (0.0133) | 0.253 | 0.136 | -0.0033 (0.0075) | +0.0045 (0.0064) | -0.0263 (0.0097) | +0.0064 (0.0041) |
| cat48 | 2.5% | - | 0.4326 | 0.4248 | 2.4% | +0.0453 (0.0124) | 0.144 | 0.275 | +0.0078 (0.0057) | +0.0164 (0.0064) | -0.0071 (0.0092) | +0.0033 (0.0037) |
| cat_other | 27.7% | - | 0.3759 | 0.3714 | 28.1% | -0.0108 (0.0034) | 0.259 | 0.160 | +0.0044 (0.0018) | +0.0039 (0.0017) | -0.0162 (0.0027) | +0.0003 (0.0010) |
| multi | 16.3% | - | 0.3843 | 0.3776 | 16.4% | -0.0189 (0.0030) | 0.102 | 0.179 | +0.0067 (0.0016) | +0.0043 (0.0016) | -0.0173 (0.0023) | -0.0005 (0.0009) |

**pos_cat_in_hist**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poscat_in_hist | 93.5% | - | 0.3932 | 0.3900 | 92.4% | +0.0060 (0.0018) | 0.154 | 0.243 | +0.0031 (0.0010) | +0.0030 (0.0010) | -0.0165 (0.0015) | +0.0009 (0.0006) |
| poscat_NOT_in_hist | 6.5% | - | 0.2846 | 0.2771 | 7.6% | -0.0872 (0.0055) | 0.427 | 0.060 | +0.0075 (0.0024) | -0.0009 (0.0018) | -0.0176 (0.0040) | -0.0031 (0.0014) |

**pos_cat_hist_share**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| phs=0 | 6.5% | - | 0.2846 | 0.2771 | 7.6% | -0.0872 (0.0055) | 0.427 | 0.060 | +0.0075 (0.0024) | -0.0009 (0.0018) | -0.0176 (0.0040) | -0.0031 (0.0014) |
| phs<=.05 | 12.0% | - | 0.3121 | 0.3068 | 13.5% | -0.0669 (0.0041) | 0.402 | 0.012 | +0.0053 (0.0021) | -0.0011 (0.0018) | -0.0252 (0.0032) | -0.0027 (0.0013) |
| phs.05-.2 | 23.0% | - | 0.3782 | 0.3804 | 23.3% | -0.0066 (0.0036) | 0.230 | 0.120 | -0.0022 (0.0021) | +0.0017 (0.0019) | -0.0214 (0.0031) | +0.0022 (0.0011) |
| phs.2-.5 | 28.9% | - | 0.4110 | 0.4085 | 27.7% | +0.0233 (0.0033) | 0.113 | 0.247 | +0.0025 (0.0020) | +0.0036 (0.0020) | -0.0154 (0.0027) | +0.0012 (0.0010) |
| phs>.5 | 29.6% | - | 0.4203 | 0.4133 | 27.9% | +0.0287 (0.0031) | 0.033 | 0.427 | +0.0071 (0.0020) | +0.0051 (0.0019) | -0.0102 (0.0026) | +0.0009 (0.0011) |

**pos_f3_catrank**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f3cat#1 | 52.5% | - | 0.4311 | 0.4274 | 48.6% | +0.0421 (0.0025) | 0.050 | 0.387 | +0.0038 (0.0015) | +0.0039 (0.0014) | -0.0056 (0.0021) | +0.0017 (0.0008) |
| f3cat#2 | 17.7% | - | 0.3923 | 0.3927 | 17.5% | -0.0035 (0.0040) | 0.185 | 0.128 | -0.0004 (0.0023) | +0.0024 (0.0023) | -0.0180 (0.0034) | +0.0017 (0.0013) |
| f3cat#3 | 9.6% | - | 0.3588 | 0.3600 | 10.1% | -0.0380 (0.0052) | 0.267 | 0.052 | -0.0011 (0.0029) | +0.0003 (0.0025) | -0.0363 (0.0041) | +0.0013 (0.0016) |
| f3cat#4-6 | 13.4% | - | 0.3020 | 0.2935 | 15.2% | -0.0802 (0.0035) | 0.383 | 0.010 | +0.0085 (0.0020) | +0.0014 (0.0018) | -0.0368 (0.0030) | -0.0037 (0.0011) |
| f3cat#7+ | 6.8% | - | 0.2274 | 0.2201 | 8.6% | -0.1037 (0.0032) | 0.520 | -0.017 | +0.0073 (0.0020) | +0.0010 (0.0018) | -0.0300 (0.0027) | -0.0035 (0.0011) |

**pos_f1_pct**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1top | 12.3% | - | 0.5563 | 0.5576 | 8.9% | +0.1011 (0.0057) | 0.082 | 0.253 | -0.0012 (0.0033) | +0.0047 (0.0032) | +0.0026 (0.0049) | +0.0045 (0.0019) |
| f1<=10% | 17.1% | - | 0.4328 | 0.4373 | 15.8% | +0.0761 (0.0045) | 0.064 | 0.335 | -0.0045 (0.0027) | +0.0018 (0.0026) | +0.0017 (0.0038) | +0.0046 (0.0014) |
| f1 10-30% | 27.3% | - | 0.3941 | 0.3926 | 27.0% | +0.0139 (0.0032) | 0.099 | 0.317 | +0.0015 (0.0020) | +0.0018 (0.0019) | -0.0084 (0.0027) | +0.0016 (0.0011) |
| f1 30-60% | 25.9% | - | 0.3358 | 0.3281 | 28.0% | -0.0447 (0.0028) | 0.200 | 0.197 | +0.0076 (0.0018) | +0.0032 (0.0017) | -0.0304 (0.0024) | -0.0018 (0.0010) |
| f1>60% | 17.4% | - | 0.2817 | 0.2703 | 20.3% | -0.1021 (0.0027) | 0.414 | 0.028 | +0.0113 (0.0017) | +0.0032 (0.0014) | -0.0404 (0.0023) | -0.0041 (0.0009) |

**pos_cold**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pos_warm | 71.0% | - | 0.3895 | 0.3808 | 70.6% | +0.0044 (0.0021) | 0.188 | 0.215 | +0.0088 (0.0013) | +0.0101 (0.0012) | -0.0088 (0.0017) | +0.0015 (0.0007) |
| pos_some_cold | 11.6% | - | 0.4080 | 0.4083 | 11.2% | +0.0036 (0.0038) | 0.079 | 0.261 | -0.0003 (0.0019) | -0.0051 (0.0019) | -0.0242 (0.0029) | -0.0016 (0.0011) |
| pos_all_cold | 17.4% | - | 0.3575 | 0.3735 | 18.2% | -0.0202 (0.0042) | 0.166 | 0.276 | -0.0159 (0.0020) | -0.0217 (0.0020) | -0.0432 (0.0031) | -0.0017 (0.0011) |

**pos_cat_share_feed**

| bucket | share val | share test | NN oof | NN tr | loss share | M-adj dev (SE) | cat-orc gap | within-orc gap | oof-tr (SE) | stk-tr (SE) | CB-oof (SE) | blend-oof (SE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pcs<=5% | 6.2% | - | 0.2398 | 0.2345 | 7.7% | -0.0803 (0.0047) | 0.579 | -0.025 | +0.0052 (0.0022) | +0.0011 (0.0022) | -0.0276 (0.0034) | -0.0032 (0.0011) |
| pcs5-15% | 19.8% | - | 0.3342 | 0.3272 | 21.5% | -0.0332 (0.0038) | 0.339 | 0.030 | +0.0070 (0.0020) | +0.0042 (0.0018) | -0.0363 (0.0029) | -0.0006 (0.0012) |
| pcs15-30% | 27.1% | - | 0.3961 | 0.3977 | 26.7% | +0.0095 (0.0034) | 0.172 | 0.164 | -0.0016 (0.0020) | +0.0024 (0.0019) | -0.0124 (0.0028) | +0.0024 (0.0010) |
| pcs30-60% | 34.2% | - | 0.4228 | 0.4193 | 32.1% | +0.0232 (0.0030) | 0.060 | 0.351 | +0.0035 (0.0018) | +0.0025 (0.0016) | -0.0078 (0.0025) | +0.0006 (0.0010) |
| pcs>60% | 12.7% | - | 0.4187 | 0.4112 | 12.0% | +0.0084 (0.0046) | 0.011 | 0.490 | +0.0074 (0.0030) | +0.0029 (0.0030) | -0.0129 (0.0041) | +0.0005 (0.0015) |
