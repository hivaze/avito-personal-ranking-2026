# review: adversarial review of the two gated-mixing studies (rulesA, gateB) and of the M-rule

Written 2026-11-15 13:42-14:10Z by the reviewer. CPU only, stored predictions only, no training, no submission file.
Work directory `nn_x/w12/mix/review/`; every criterion is in `DECLARATION.txt` with its time stamp before the number it judges.
Marks: [M] measured here, [D] read in a file, [I] inferred. ± = user-clustered SE of the per-feed paired difference (equal to the per-feed SE to 5 decimals here: 14,398 groups for 14,423 feeds). Val = 14,423 feeds.
Compositions: C1 = FD16 + 4 wide full-data popularity-light students (the FD16P20 file), C2 = FD8 + 4 (the FD8P20 file), C3 = W16 + 4 (held-out twins), C4 = W8 + 4.
PLAIN = weight 0.2 everywhere, w_ft 0.5. M-rule = 0.2 on M < 50, 0.4 on M >= 50, w_ft 0.5.

## 0. Verdict

1. **confirmed = FALSE.** Neither searcher names a new candidate, and none exists: nothing other than PLAIN and the standing M-rule passes my bars. Both searchers' code is correctly nested (user-grouped folds, user 0 as singletons), their baselines and every in-sample number reproduce to 1e-5 with my own loader, z and NDCG, and their nested numbers reproduce with fold seeds they did not use (largest gap 0.00017).
2. **Their "no candidate" stands and is stronger than they state.** The placebo both used (same-recipe students in the popularity-light role) carries 0.58-0.72 of the per-feed noise of the real pools, so the p-values drawn from it are too small. Under a correctly scaled null (user-level sign-flip, 400 draws) the row A singled out, F(Hq), has p 0.06-0.13 alone and family-wise p >= 0.16 (A: 0.010 / 0.030), and fails on the twins with my folds too (-0.00009, -0.00034).
3. **The M-rule is NOT SHOWN on the full-data members the final files are made of.** Pre-declared headline (mean of the nested refit and the cross-member-set nested read, 10 own fold seeds): FD16 + 4 **+0.00002 ± 0.00027** (sign-flip p 0.42), FD8 + 4 +0.00015 ± 0.00028 (p 0.23), W16 + 4 **+0.00030 ± 0.00026** (p 0.25), W8 + 4 +0.00015 ± 0.00028 (p 0.45). The in-sample readings on record (+0.00026, +0.00051, +0.00071, +0.00043) are 2.4 to 13 times larger.
4. **A's sentence "the M-rule family is real" does not survive.** By A's primary criterion the family "is not shown" (nested +0.00018 ± 0.00030 on C1, p 0.19; A says so). "Real" rests on the two-member-set statistic of A's addendum 3 (designed after the first tables), whose p 0.040 comes from the understated placebo; the sign-flip p of the same statistic is 0.13, above that addendum's own 0.05 bar. Correct wording: supported, not shown.
5. **On full-data members the gate itself is not doing anything.** "0.4 minus 0.2" on long feeds minus the same on short feeds: -0.00008 ± 0.00068 (FD16), +0.00087 ± 0.00069 (FD8), against +0.00239 ± 0.00062 and +0.00194 ± 0.00063 on the twins. The M-rule equals a random gate of the same size on FD16 (p 0.55). What full-data members show is a dose effect: 0.4 everywhere reads +0.00048 ± 0.00032 / +0.00050 ± 0.00032 in-sample, nested +0.00039 ± 0.00031 / +0.00040 ± 0.00032 (p 0.16 / 0.24), and is negative on the twins. Not a candidate.
6. **The lead's reading rule for the public pair is too lenient as a "+2 SE" claim.** With the in-sample val leg (+0.00051 ± 0.00021) any D_pub >= +0.00004 passes. With an honest val leg the FD8 pair needs D_pub >= +0.0004 (selection-adjusted leg) to +0.0006 (nested leg), the FD16 pair +0.00065 to +0.0008. As an expected-value decision the rule is defensible (section 5.4); as evidence it is not.
7. **Transfer is clean.** Gate cells cover the same range on val, public-like and private-like feeds; the rule moves test rankings exactly as much as val rankings; pool geometry is identical. The rule's weakness is evidence, not transfer.
8. Best single number for the files with the owner: M-rule minus plain about **+0.0002 on val (range 0.0000 to +0.0004), total SE about 0.0003**. P(M-rule file beats its plain twin on public) about 0.6 for the FD16 pair and about 0.7 for the FD8 pair [I].

## 1. Check 1: reproduction (PASS)

Own code path (`rlib.py`): polars read of the feed table, own float64 per-feed z, own NDCG (stable sort with negatives first among ties), own grid tensor. Project functions were imported only as the reference.

| item | mine [M] | on record [D] |
|---|---|---|
| own NDCG vs `src/metrics.ndcg_per_feed`, same scores | max abs per-feed difference 0.0 (also on heavily tied random scores) | |
| FD16P20 / FD16P20@m0.2_0.4 | 0.402714 / 0.402975 (project path identical) | 0.40271 / 0.40298 |
| FD8P20 / @m | 0.402396 / 0.402904 | 0.40240 / 0.40290 |
| W16P20w / @m | 0.402141 / 0.402847 | 0.40214 / 0.40285 |
| rule minus plain C1 / C2 / C3 / C4 | +0.00026 ± 0.00021 / +0.00051 ± 0.00021 / +0.00071 ± 0.00021 / +0.00043 ± 0.00019 | same |
| A in-sample on C1: constant, C(M50), F(Hq), G(M3q), K(Mq,Hq) | +0.00054, +0.00053, +0.00076, +0.00070, +0.00083 | same |
| B in-sample: best constant C1 (0.40, 0.70), C3 (0.40, 0.40); MSTEP C1 / C3 | +0.00054, +0.00044; +0.00067 / +0.00083 | same |
| B dose contrast 0.4 minus 0.2: all, M < 50, M >= 50 (C1 \| C3) | +48, +53, +44 \| +22, -119, +119 (1e-5) | same |
| own-path TEST score vs the stored branches of both built FD16 files | max abs difference 6e-7 | |

The last row means the two FD16 files in the slots are exactly the form `build_r4.py` describes.

## 2. Check 2: nesting

### 2.1 Code audit

- Folds: both use user-grouped folds with every user-0 feed its own group (`rulesA/lib.py folds`, `gateB/lib.py groups` + `lib2.py folds`). B's inner folds, winsor bounds, standardisation and stump cuts are computed on the training part. A's rule choice uses training sums only. **No leak found.**
- Choices made with labels outside every nested loop:
  - The cut 50 and the weight 0.4 (both studies' fixed comparator; A's gate M50; B's MSTEP cut list 25 / 50 / 100). They come from a full-val table and a 75-rule scan on other models of the same val feeds (`premortem.md` 3.2-3.3; that scan's grid ended at 0.4).
  - A: the row reported as "clears the primary bars" is the top of 95 families by nested gain on C1. A handles it with a family-wise placebo p, which is understated (section 3.2).
  - A: the two-member-set statistic T2, its matched placebo, and the 8-composition summary were designed after the first nested tables were seen (A says so). The 8 compositions share all feeds and 20 of 24 members.
  - B: "best single family on FD16 + 4" is picked from the nested table of 8 families. B rejects it on replication.
- Not a leak, worth knowing: the fine-tuned branch on val is out-of-fold over the fine-tune's own user folds, which differ from the mixing folds. It affects every rule equally.

### 2.2 Nested re-run with my folds (seeds 7001-7010, 10 x 5 folds), all minus PLAIN

| family | C1 mine / theirs | C2 | C3 | C4 |
|---|---|---|---|---|
| A constant (w_pop, w_ft) | +0.00039 / +0.00027 | +0.00033 / +0.00031 | +0.00008 / +0.00006 | -0.00020 / -0.00026 |
| C(M50) = M-rule family, A grid | +0.00008 ± 0.00028 / +0.00018 | +0.00018 ± 0.00024 / +0.00021 | +0.00064 ± 0.00025 / +0.00055 | +0.00025 ± 0.00028 / +0.00028 |
| C(M70) | +0.00008 / +0.00023 | +0.00007 / +0.00007 | +0.00033 / +0.00026 | +0.00034 / +0.00044 |
| F(Hq) | +0.00073 ± 0.00032 / +0.00076 | +0.00048 / +0.00041 | -0.00009 / -0.00019 | -0.00034 / -0.00034 |
| G(M3q) | +0.00031 / +0.00031 | +0.00003 / +0.00010 | +0.00043 / +0.00044 | +0.00026 / +0.00028 |
| K(M70, Hq) | +0.00042 / +0.00047 | +0.00023 / +0.00028 | +0.00021 / +0.00018 | +0.00047 / +0.00038 |
| B CONSTp | +0.00039 / +0.00039 | | -0.00015 / -0.00023 | |
| B CONST2 | +0.00027 / +0.00017 | | -0.00014 / -0.00000 | |
| B MSTEP (cuts 25 / 50 / 100) | **+0.00007 / +0.00024** | | +0.00050 / +0.00043 | |
| H stump (mine, H only) / B STUMP1 (8 features) | +0.00057 ± 0.00030 / +0.00065 | | -0.00047 / -0.00038 | |

All within my 0.00015 allowance except B's MSTEP on C1 (0.00017): fold-seed sensitive, my range over 10 seeds is -0.00043 to +0.00037. In-sample numbers next to these are in `log_step2.txt`.
Not re-run: B's AUTO and the three LIN families (B's own verdict on them is negative), A's families with per-branch weights and the U0 / cold-share gates.

## 3. Check 3: selection bias

### 3.1 My placebo: user-level sign-flip null

One random sign per user group flips every contrast to PLAIN jointly, with the same signs on all compositions. Every rule then has expected gain 0, while the correlation between rules, between member sets, and the heavy tails are kept. 400 draws, identical search code.
Sanity of my own code: worst |z| of a null mean over 56 nested statistics is 2.22 (bar 3). A nested read is 0 under the null.

Selection optimism under this null (C1): the in-sample best of a 2-parameter family reads +0.00042 to +0.00058 on average (q95 +0.0009 to +0.0012); a single H stump +0.00104 (q95 +0.00165). A's "in-sample scan numbers below about +0.0008 are not evidence" should read **below about +0.0013**.

| statistic | observed | sign-flip p | searcher's p |
|---|---|---|---|
| F(Hq) nested on C1 (5 fold seeds) | +0.00073 | 0.129 | 0.010 (A) |
| the same, family-wise over 6 of A's families / my 12 | | 0.208 / 0.272 (lower bounds for A's 95) | 0.030 (A) |
| F(Hq) minus the M-rule family, mean of C1..C4 | +0.00001 | 0.48 | |
| C(M50) nested on C1 | +0.00008 to +0.00018 | 0.33 to 0.42 | 0.19 (A) |
| C(M50) nested on C3 | +0.00064 | 0.08 | 0.040 (A) |
| C(M50), mean of C1 and C3 (A's T2) | +0.00036 ± 0.00022 (10 seeds) | 0.13 | 0.040 (A) |
| C(M50), mean of C1..C4 | +0.00029 | 0.155 | |
| best nested row of mine on any composition, family-wise | +0.00076 | 0.27 | |

### 3.2 Why the searchers' placebo is too narrow (MAJOR)

Both used "4 same-recipe wide students play the popularity-light pool". Those pools correlate 0.9955 with their base; the real popularity-light pool correlates 0.985. Measured on B's 4 structured splits of each base set: the per-feed sd of the M-rule contrast is 0.0144-0.0185 against 0.0256 / 0.0248 for the real pools (ratio 0.58-0.72). A's stored null (`pl_PLA_FD16.npz`) has sd 0.00020-0.00023 for nested family gains; the sign-flip null has 0.00041-0.00051 with the same 5-seed averaging.
Consequences: A's p_A and p2 and B's "above 44 of 44, z +7.9" (D1) are anti-conservative. B's feature-permutation placebo and A's gate-permutation placebo are valid for their own question (does the gate carry information given these feeds) and are not affected.

### 3.3 The M-rule's own selection

The rule came out of a 75-rule scan (3 gates x 25 weight pairs) on the twins. Emulated under the null:

| reading | observed | null | p |
|---|---|---|---|
| in-sample winner of the 75-rule scan on C3 (it IS 0.2 / 0.4 by M >= 50) | +0.00071 | winner mean +0.00042, q95 +0.00076 | 0.090 |
| fixed rule, mean of C1..C4, against the scan maximum on that mean | +0.00048 | q95 +0.00055 | 0.110 |
| winner selected on C3, read on C1 (other models, same feeds) | +0.00026 | transferred mean +0.00011, sd 0.00027 | 0.27 |
| the same, read on C2 | +0.00051 | mean +0.00012, sd 0.00027 | 0.08 |
| winner selected on mean(C3, C4), read on mean(C1, C2) | +0.00038 | mean +0.00014, sd 0.00022 | 0.14 |
| mirror: winner on mean(C1, C2) is 0.4 / 0.4 (a constant), read on the twins | +0.00007 | mean +0.00015 | 0.61 |

"Other trained models, same val feeds" carries 27% of the selection optimism (per-feed correlation of the rule contrast between member sets on long feeds: 0.30 for C1~C3, 0.57 for C1~C2, 0.20 between member-disjoint halves). The fixed rule's reads on full-data members shrink to +0.00015 (C1) and +0.00039 (C2) after subtracting it.

### 3.4 Multiplicity

- Pre-mortem: 75 gated rules x 5 compositions, about 20 constant-weight looks. Lead: about 6 dose rules.
- A: 131,547 rules in 100 families (95 eligible), 3 selection variants, 8 compositions, 9 fixed hypotheses, dose tables, 6 post hoc contrasts.
- B: 8 families + AUTO; about 1.8 million enumerable stump rules, product stumps beyond that, 3 continuous families x 5 ridge strengths, 3 combiners, an in-sample oracle map of 58 cells x 4 contrasts x 2 compositions.
- Mine: 12 families, 250,469 rules, 8 compositions (4 + 4 member halves), the 75-rule emulation.

Plainly: across roughly two million enumerated rules, the only structure that appears in more than one member set is the M step, and it is family-wise significant on none (best: p 0.08-0.09 on the twins, where it was found).

## 4. Check 4: transfer (label-free on test)

| | val | public-like (8,562) | private-like (20,033) |
|---|---|---|---|
| share of feeds with M >= 50 | 0.591 | 0.613 | 0.608 |
| KS distance of M inside the long / short cell vs val | | 0.040 / 0.037 | 0.024 / 0.022 |
| range of M | 5-299 | 5-300 | 5-300 |
| share with H >= 190 | 0.504 | 0.502 | 0.499 |
| FD16 pair: top-1 differs from plain, long feeds | 7.2% | 7.5% (392 feeds) | 7.0% (854 feeds) |
| FD8 pair: top-1 differs, long feeds | 7.3% | 8.2% (431) | 7.4% (896) |
| per-feed correlation of the two pools, train-only branch, long feeds | 0.9849 | 0.9843 | 0.9843 |
| the same, fine-tuned branch | 0.9887 | 0.9886 | 0.9886 |

T1 PASS. In total 1,246 of 28,595 test feeds change top-1 for the FD16 pair and 1,327 for the FD8 pair. The whole val gain of the rule on C1 sits in the 616 val feeds whose top-1 changes (+0.0056 ± 0.0044 inside them): the public pair is decided by about 400 feeds.

Concentration of the val gain (long feeds, by the pmix cold-share bin):

| cold share | share of feeds val / public / private | C1 gain per feed | C2 | C3 | C4 |
|---|---|---|---|---|---|
| < 15% | 0.036 / 0.001 / 0.000 | +0.00160 ± 0.00138 | +0.00090 | +0.00258 | -0.00088 |
| 15-30% | 0.305 / 0.142 / 0.078 | +0.00057 ± 0.00045 | +0.00071 | +0.00116 | +0.00055 |
| 30-45% | 0.217 / 0.358 / 0.350 | -0.00021 ± 0.00065 | +0.00085 | +0.00071 | +0.00088 |
| >= 45% | 0.033 / 0.111 / 0.180 | +0.00226 ± 0.00190 | +0.00222 | +0.00314 | +0.00296 |

My declared flag is not raised on any composition (pmix-private stays above half the val gain: C1 +0.00038 ± 0.00036, C2 +0.00072 ± 0.00034). Two cautions for the FD16 file only: its private-mix read rests on the 480 val feeds of the >= 45% cell, and its gain is on the first val day only (+0.00052 ± 0.00030 against -0.00002 ± 0.00030). C2, C3, C4 show neither pattern, so I read both as noise of one composition.

## 5. Check 5: the M-rule estimate

### 5.1 Estimators (10 own fold seeds; sign-flip p from 400 draws)

| composition | fixed rule, in-sample | nested refit, B grid / A grid | cut also free (6 cuts) | cross-member-set nested | HEADLINE (declared) | p | fixed rule minus transferred selection optimism | p |
|---|---|---|---|---|---|---|---|---|
| C1 FD16 + 4 | +0.00026 ± 0.00021 | -0.00011 ± 0.00029 / +0.00008 ± 0.00028 | +0.00005 | +0.00015 ± 0.00026 | **+0.00002 ± 0.00027** | 0.42 | +0.00015 | 0.27 |
| C2 FD8 + 4 | +0.00051 ± 0.00021 | +0.00001 ± 0.00033 / +0.00018 ± 0.00024 | +0.00022 | +0.00030 ± 0.00028 | **+0.00015 ± 0.00028** | 0.23 | +0.00039 | 0.08 |
| C3 W16 + 4 | +0.00071 ± 0.00021 | +0.00050 ± 0.00026 / +0.00064 ± 0.00025 | +0.00050 | +0.00010 ± 0.00029 | **+0.00030 ± 0.00026** | 0.25 | (selected here) | 0.09 |
| C4 W8 + 4 | +0.00043 ± 0.00019 | +0.00041 ± 0.00028 / +0.00025 ± 0.00028 | +0.00034 | -0.00011 ± 0.00032 | **+0.00015 ± 0.00028** | 0.45 | | |

Headline at the public / private mix: C1 +0.00015 ± 0.00036 / +0.00030 ± 0.00045; C2 +0.00030 / +0.00042; C3 +0.00046 / +0.00063.
By my declared statements the rule is NOT SHOWN on C1 and NOT SHOWN on C3. The headline is conservative for a fixed rule (it pays for refitting 289 weight pairs); the last column is the efficient read. Together: about +0.0002 on the full-data files, total SE about 0.0003.

### 5.2 Member-disjoint halves (my design; fixed rule, 2 popularity-light students each)

FD members s951-954 + 2 students: +0.00005 ± 0.00022. FD members s955-958 + the other 2: +0.00042 ± 0.00020. Twins: +0.00020 ± 0.00020 and +0.00038 ± 0.00021.
Mean of the four +0.00026 ± 0.00013 (sign-flip p 0.032 without selection); the two full-data halves +0.00023 ± 0.00016 (p 0.07). All four positive, a factor 8 apart inside the full-data pair: member-combination noise is as large as the effect.

### 5.3 What the rule is made of

| | 0.4 minus 0.2 on M >= 50 | on M < 50 | interaction (z) | M-rule vs random gate of equal size, p | M-rule minus constant 0.4 |
|---|---|---|---|---|---|
| C1 | +0.00044 ± 0.00036 | +0.00053 ± 0.00058 | -0.00008 ± 0.00068 (-0.1) | 0.55 | -0.00022 ± 0.00024 |
| C2 | +0.00086 ± 0.00035 | -0.00001 ± 0.00059 | +0.00087 ± 0.00069 (+1.3) | 0.10 | +0.00000 ± 0.00024 |
| C3 | +0.00119 ± 0.00035 | -0.00119 ± 0.00051 | +0.00239 ± 0.00062 (+3.8) | < 0.001 | +0.00049 ± 0.00021 |
| C4 | +0.00072 ± 0.00032 | -0.00122 ± 0.00055 | +0.00194 ± 0.00063 (+3.1) | 0.001 | +0.00050 ± 0.00022 |

The long-feed part is positive in all 8 member sets I built (+0.00008 to +0.00119 per long feed). The short-feed part (keep 0.2) is supported only by the twins; on short feeds full-data members differ from their twins by +0.00172 ± 0.00067 (16-base), +0.00227 ± 0.00079 and +0.00068 ± 0.00075 in the two member-disjoint halves. A flagged this as post hoc (largest of 6 contrasts); it holds in both halves. I have no mechanism for it; the twins use a holdout-selected checkpoint and the full-data members the last step [D].
Practical reading: the M-rule never reads below PLAIN (8 of 8 member sets), while constant 0.4 loses 0.0005 to it on the twins. The M-rule is the safer of the two forms; the evidence for it on the final members is the dose, not the gate.

### 5.4 The public pair

Paired public SE 0.00027-0.00028 [M]. P(D_pub > 0) [I]: FD16 pair 0.52 (headline) to 0.65 (selection-adjusted read); FD8 pair 0.65 to 0.85. Central 0.6 and 0.7, as the lead and A have it.

The declared reading rule ("D_pub >= 0 and inverse-variance mean of the val delta and D_pub >= +2 SE"), by val leg:

| val leg | FD8 pair: D_pub needed | FD16 pair |
|---|---|---|
| in-sample (as declared): +0.00051 ± 0.00021 / +0.00026 ± 0.00021 | +0.00004 | +0.0005 |
| selection-adjusted: +0.00039 ± 0.00027 / +0.00015 ± 0.00027 | +0.00037 | +0.00065 |
| nested headline: +0.00015 ± 0.00028 / +0.00002 ± 0.00027 | +0.00061 | +0.00079 |

D_pub > 0 alone is weak evidence: it happens with probability 0.5 if the true effect is 0 and 0.87 if it is +0.0003. As a decision under a prior of +0.0002 ± 0.0003, adopting has non-negative expected value whenever D_pub > about -0.00015, so "adopt at D_pub >= 0" is a reasonable choice; it should not be recorded as a +2 SE result unless D_pub >= about +0.0004 [I]. The two pairs are not two tests: their contrasts correlate 0.57, so a second pair cuts the SE by 12%.

## 6. Issues

- MAJOR: both searchers' member-role placebo carries 0.58-0.72 of the real per-feed noise; A's p 0.010 / 0.030 (F(Hq)), 0.040 (M-rule family across two member sets), 0.19 (C1) and B's "z +7.9" are anti-conservative. Corrected: 0.06-0.13 / >= 0.16, 0.13, 0.33-0.42.
- MAJOR: A's verdict item 4 ("the M-rule family is real") goes beyond A's primary criterion ("is not shown") on the strength of a post hoc statistic whose p is understated (0.040; 0.13 under the correct null). On the full-data members the nested refit is -0.00011 to +0.00018.
- MAJOR: on full-data members the gate is not specific (interaction -0.1 SE on FD16, +1.3 SE on FD8; random gate p 0.55 / 0.10). The short-feed half of the rule is twin-only evidence.
- MAJOR: the reading rule for the public pair combines D_pub with an in-sample, selection-optimistic val leg; any D_pub >= +0.00004 would be booked as a pooled +2 SE.
- MEDIUM: "7 of 8 compositions positive" and "all six / seven compositions positive" are not independent replications (same feeds, 20 of 24 members; contrast correlation 0.3-0.6). Member-disjoint halves read +0.00005 and +0.00042.
- MEDIUM: cut 50 and weight 0.4 were label-prompted and sit outside every nested loop in both studies (fixed comparator, A's M50 gate, B's cut list).
- MEDIUM: B's nested MSTEP on FD16 + 4 (+0.00024) is fold-seed sensitive: +0.00007 with my 10 seeds, range -0.00043 to +0.00037.
- MEDIUM: A's "in-sample numbers below +0.0008 are not evidence" is too low; +0.0013 for a single small family under the correct null.
- MEDIUM: FD16 file only: gain on the first val day only, and private-mix read resting on 480 val feeds. Not seen on the other three compositions.
- LOW: A's ± are per-feed SE without selection variance (stated by A); the true spread of a nested family gain is 0.0004-0.0005, not 0.0003.
- LOW: my H stump uses all-val quantiles of H (label-free) instead of training-part quantiles; B's AUTO / LIN families and A's per-branch families were not re-run.
- No BLOCKER: no leak, no wrong baseline, no error that changes "no new candidate".

## 7. Files

`nn_x/w12/mix/review/`: `DECLARATION.txt` (blocks 0-10), `PROGRESS.md`, `rlib.py`, `step1.py` (reproduction), `step2.py` (nested, estimates, specificity), `placebo.py` + `placebo_read.py` (sign-flip null), `extra.py` (P6, V1, V2), `transfer.py`; logs `log_step1.txt`, `log_step2.txt`, `log_placebo_read.txt`, `log_extra.txt`, `log_transfer.txt`; `cache/` (pool arrays and grid tensors, not submissions).
Commands: `cd nn_x/w12/mix/review && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 NUMBA_NUM_THREADS=12 ../../../../.venv/bin/python step1.py` (12 s), then `step2.py` (5 s), `placebo.py 0 200 a` and `placebo.py 200 200 b` (2 min each), `placebo_read.py`, `extra.py` (30 s), `transfer.py` (5 s).
Compliance: CPU only, at most 2 processes x 12 threads, largest arrays 4 x 2.9M float64 rows (memory not measured, far below the 30 GB cap [I]), nothing written outside the review directory and this report, no process signalled, no network.
