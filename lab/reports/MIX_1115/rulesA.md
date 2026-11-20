# rulesA: declared rule families for the two mixing weights (searcher A), 2026-11-15

Directory: `nn_x/w12/mix/rulesA/` (DECLARATION.txt, PROGRESS.md, scripts, logs, result files). CPU only. No file under subs/ or preds/ was written, nothing was submitted.
Marks: [M] measured here, [D] read in a file, [I] inferred. All deltas are final val NDCG (14,423 feeds), paired per feed, "+-" = paired per-feed SE.
Compositions: C1 = FD16 + 4 wide full-data popularity-light (FD16P20, PRIMARY); C2 = FD8 + 4 (FD8P20); C3 = W16 + 4 wide held-out (W16P20w); C4 = W8 + 4 (W8P20w). C3 / C4 are other trained models on the SAME val feeds.
PLAIN = w_pop 0.2, w_ft 0.5. M-RULE = w_pop 0.2 on M < 50, 0.4 on M >= 50, w_ft 0.5.

## 0. Verdict

1. **No small gated rule beats the current forms.** 0 of 95 eligible families (about 131,000 rules, 3 selection variants, 4 compositions) pass the criteria declared before the first labelled number. No family is above the M-rule family (nested) on all 4 compositions; none is above the fixed M-rule on all 4 [M].
2. The only rows that clear the primary-composition bars are three "w_ft by history length" forms (F(Hq): w_pop 0.4, w_ft 0.7 on H < 190, 0.5 else): nested +0.00076 +- 0.00032, placebo p 0.010, family-wise p 0.030 on C1. They do **not replicate**: -0.00019 / -0.00034 on the held-out twins, the rule fitted on the twins reads -0.00005 on C1, across two member sets they are below the fixed M-rule (-0.00020 +- 0.00019), and on the same FD16 base with the 8-member popularity-light pool the family reads -0.00054 +- 0.00022. The fitted w_ft step changes sign between member sets (0.7 | 0.5, 0.7 | 0.8, 0.4 | 0.6, 0.6 | 0.3 on C1..C4). Member-combination noise [M].
3. **The M-rule family is real but smaller than its in-sample reading.** Weights refitted inside training folds: +0.00018 +- 0.00030 (C1), +0.00021 +- 0.00025 (C2), +0.00055 +- 0.00026 (C3), +0.00028 +- 0.00029 (C4). Alone on C1 it is not shown (placebo p 0.19). Mean over the two 16-base member sets: **+0.00036 +- 0.00022, matched-placebo p 0.040**; at the public / private mix +0.00049 +- 0.00030 / +0.00065 +- 0.00039. The fixed 0.2 / 0.4 rule reads +0.00026 / +0.00051 / +0.00071 / +0.00043: about 0.0001-0.0002 of that is selection optimism (placebo: +0.00021 for a 2-parameter fit). With the 8-member popularity-light pools (4 wide + 4 d128; addendum 6) the nested family reads -0.00011 +- 0.00027 (FD16), +0.00028 +- 0.00022 (FD8), +0.00074 +- 0.00027 (W16), +0.00009 +- 0.00022 (W8) and the fixed rule -0.00010 / +0.00047 / +0.00064 / +0.00037. Over all 8 compositions: nested family mean +0.00028 +- 0.00017 (7 of 8 positive), fixed rule mean +0.00041 +- 0.00014 (7 of 8); **the two FD16-base compositions are the weakest (nested +0.00018, -0.00011; fixed +0.00026, -0.00010)** [M].
4. **Refitting the weights does not help**: the fixed (0.2, 0.4) is at or above the nested refit on all 4 compositions (+0.00008, +0.00030, +0.00016, +0.00015; 7 of 8 with the 8-member pools, mean +0.00013 +- 0.00007). The long-feed half of the rule (0.4 against 0.2 on M >= 50) is positive in all 8 base x popularity-light pool crossings (+0.00044 .. +0.00119 per long feed) [M].
5. Declared hypotheses: h1 (w_ft 0.7 on cold feeds) NOT SHOWN; h2 (0.5 on long feeds with 8 popularity-light members) REFUTED by the declared rule (0.5 minus 0.4: -0.00020 +- 0.00014 on W16 + 8, -0.00017 on W8 + 8, -0.00019 on FD8 + 8; +0.00013 +- 0.00014 on FD16 + 8, where 0.4 itself is -0.00010 against PLAIN; 8 = 4 wide + 4 d128, the 8 wide full-data students were not finished); h3: no single-branch form replaces the rule (train-only branch carries 0.45-0.97 of the gain, fine-tuned 0.26-0.55); h4 (three-step) NOT SHOWN [M].
6. w_ft = 0.5 stands on val at every mix (no constant 0.6 / 0.7 reads +1 SE at the private mix on C1; negative on C3) [M].
7. Selection bias, measured: a greedy scan of the whole rule set on a placebo with nothing to find reads **+0.00057 in-sample (q95 +0.00082)**; the top row of a nested table of 95 families reads +0.00038 (q95 +0.00068). In-sample scan numbers below about +0.0008 are not evidence [M].
8. Recommendation: keep 0.2 / 0.4 by M >= 50 and w_ft 0.5; spend no public slot on a finer rule. For the lead's reading rule of the public pair, the val leg should be the nested number (+0.00036 +- 0.00022), not the in-sample one: with it the pair needs D_pub >= about +0.0003 for a pooled +2 SE; with the in-sample leg any D_pub >= 0 passes [I].

## 1. Reproduction (step 1) [M]

Own loader (member files -> per-feed z -> 4 pools -> branches -> final) through the project's `feed_rank_z` / `ndcg_per_feed`, and an own numba kernel that returns the per-feed NDCG for all 294 weight combinations (7 x 7 x 6). Declared bar: levels equal to the record at 5 decimals, kernel equal to the reference path to 1e-5. **PASS** (the kernel agrees on every feed).

```
labels: unique [0 1] rows 1389971 feeds 14423
C1 FD16+FDP: plain 0.402714 (record 0.40271) | M-rule 0.402975 (record 0.40298) | rule-plain +0.00026+-0.00021 (record +0.00026+-0.00021) | levels OK delta OK | kernel: plain 0.402714 M-rule 0.402975 max|mean diff| 0.00e+00 mean|per-feed diff| 1.94e-18 feeds differing 0 OK | 6s
C2 FD8+FDP: plain 0.402396 (record 0.4024) | M-rule 0.402904 (record 0.4029) | rule-plain +0.00051+-0.00021 (record +0.00051+-0.00021) | levels OK delta OK | kernel: plain 0.402396 M-rule 0.402904 max|mean diff| 0.00e+00 mean|per-feed diff| 2.04e-18 feeds differing 0 OK | 7s
C3 W16+P1w: plain 0.402141 (record 0.40214) | M-rule 0.402847 (record 0.40285) | rule-plain +0.00071+-0.00021 (record +0.00071+-0.00021) | levels OK delta OK | kernel: plain 0.402141 M-rule 0.402847 max|mean diff| 0.00e+00 mean|per-feed diff| 2.00e-18 feeds differing 0 OK | 9s
C4 W8+P1w: plain 0.402183 (record None) | M-rule 0.402610 (record None) | rule-plain +0.00043+-0.00019 (record +0.00043+-0.00019) | levels OK delta OK | kernel: plain 0.402183 M-rule 0.402610 max|mean diff| 0.00e+00 mean|per-feed diff| 2.09e-18 feeds differing 0 OK | 11s
STEP 1 PASS
```

Unit test of the nested search against an independent brute-force implementation (rules written out from the feed properties), 7 families: exact agreement.

```
C(M50)       rules    49 | brute in-sample +0.000533 nested +0.000181 | lib +0.000533 +0.000181 | max abs diff 1.1e-16 OK
F(Hq)        rules   252 | brute in-sample +0.000762 nested +0.000762 | lib +0.000762 +0.000762 | max abs diff 1.1e-16 OK
I(M50)       rules  2401 | brute in-sample +0.000587 nested +0.000072 | lib +0.000587 +0.000072 | max abs diff 1.7e-16 OK
J(Mq,CS45)   rules  1764 | brute in-sample +0.000543 nested -0.000097 | lib +0.000543 -0.000097 | max abs diff 1.1e-16 OK
J(CS45,Mq)   rules  1764 | brute in-sample +0.000576 nested +0.000029 | lib +0.000576 +0.000029 | max abs diff 1.1e-16 OK
K(Mq,Hq)     rules  2401 | brute in-sample +0.000833 nested +0.000472 | lib +0.000833 +0.000472 | max abs diff 2.2e-16 OK
G(M3r)       rules   343 | brute in-sample +0.000668 nested +0.000333 | lib +0.000668 +0.000333 | max abs diff 5.6e-17 OK
UNIT TEST PASS
```

## 2. What was declared before any labelled number (DECLARATION.txt, 13:16Z; full text in the appendix)

- Weight grids: w_pop in {0, 0.1, .., 0.6}, w_ft in {0.3, .., 0.8}.
- Gates, thresholds fixed label-free (pooled val + test quantiles, or cuts already on record): M50 (M >= 50), Mq (M >= 70), CS45 (segment cold share >= 0.45; the pmix variable), CLq (it_hist_all_luo == 0 share >= 0.4225), Hq (H >= 190), U0, DAY (diagnostic only: test lies after both val days); three-cell M3q (39 / 117), M3r (30 / 100), CL3q, H3q.
- 12 forms, 100 families (95 eligible), <= 4 free weights each: constant (A, B), w_pop by one gate (C; C(M50) = the M-rule's family), + free constant w_ft (D), w_ft by one gate (E, F), three-cell w_pop (G, H), per-branch w_pop by a gate (I), w_pop by one gate x w_ft by another (J), w_pop on 2 x 2 cells (K), three-cell w_ft (L).
- Nested read: 5 user-grouped folds x 5 fold seeds (user_id 0 feeds are singleton groups; no user straddles folds, asserted). Variant A = best rule of a family on the training part; B-all / B-3 = best family by inner cross-validation (all eligible / <= 3 parameters); G = greedy best rule over everything.
- Placebo PL-A (pseudo popularity-light pool = 4 random of the 16 wide students against the other 12, fine-tuned / train-only roles swapped on a random half): 200 draws on FD16, 100 on W16; PL-G (gates permuted across feeds, real pools): 200.
- Candidate criteria c1-c5 (primary composition; nested >= +2 SE and p_A <= 0.05; above the M-rule family beyond the placebo q95 and above the fixed M-rule; replication on >= 2 of 3 other compositions and > PLAIN on all 4; pmix; test coverage of every gate cell >= 5%).
- Two cold-share definitions exist and differ (correlation 0.80): the segment one used by pmix (item never seen in train; >= 0.45 on 8.4% of val, 21.5% of public-like, 32.7% of private-like feeds) and the task's it_hist_all_luo == 0 share (25.9% / 41% / 51%). Both were used (CS, CL).
- Addenda written during the work, each before its own numbers: 1 cross-model read; 2 leak check (a stop rule tripped, see section 4); 3 two-member-set statistic with a matched placebo; 4 w_ft curve under pmix weights; 5 one decomposition of a post hoc contrast; 6 the same nested search on four 8-member-pool compositions (the full-data d128 fine-tunes arrived at 13:35Z). Addenda are post hoc in design and are marked so below.

Multiplicity: 100 families, 131,547 rules, 3 selection variants, 4 compositions (+ 4 in addendum 6), 2 baselines; 9 fixed hypothesis rules; 6 addenda.

## 3. Nested read (step 3) [M]

Columns: in-sample = best rule on all val minus PLAIN; nested = out-of-fold; p_A = placebo p of the nested gain (PL-A, 200 draws); p_A,M = placebo p of "family minus C(M50)"; p_G = gate-permutation p of "family minus A"; "twins->C1" = rule fitted on the training users of C3, read on the held-out users of C1 (addendum 1); T2 = mean of the nested gains on C1 and C3, p2 from the matched placebo (addendum 3, 100 draws). The paired SE does not contain the selection variance; the placebo columns do.

| family | par | rules | C1 in-sample | C1 nested - plain (z) | p_A | C1 nested - fixed M-rule | C1 nested - nested C(M50) | p_A,M | p_G | C2 nested - plain | C3 nested - plain (p_A W16) | C4 nested - plain | twins->C1 other models + other users | T2 = mean(C1, C3) | p2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F(Hq) | 3 | 252 | +0.00076 | +0.00076 +- 0.00032 (+2.4) | 0.010 | +0.00050+-0.00027 | +0.00058+-0.00023 | 0.005 | 0.010 | +0.00041+-0.00032 | -0.00019+-0.00028 (0.861) | -0.00034+-0.00022 | -0.00005+-0.00029 | +0.00029 +- 0.00024 | 0.059 |
| J(U0,Hq) | 4 | 1764 | +0.00083 | +0.00072 +- 0.00031 (+2.3) | 0.010 | +0.00045+-0.00026 | +0.00054+-0.00023 | 0.005 | 0.025 | +0.00032+-0.00032 | -0.00013+-0.00028 (0.802) | -0.00030+-0.00020 | +0.00004+-0.00029 | +0.00029 +- 0.00023 | 0.059 |
| J(CS45,Hq) | 4 | 1764 | +0.00078 | +0.00066 +- 0.00032 (+2.0) | 0.010 | +0.00039+-0.00027 | +0.00047+-0.00023 | 0.005 | 0.020 | +0.00024+-0.00031 | +0.00021+-0.00030 (0.317) | -0.00036+-0.00020 | +0.00009+-0.00032 | +0.00043 +- 0.00024 | 0.010 |
| J(Hq,Hq) | 4 | 1764 | +0.00082 | +0.00051 +- 0.00028 (+1.8) | 0.030 | +0.00025+-0.00027 | +0.00033+-0.00027 | 0.050 | 0.085 | +0.00027+-0.00032 | -0.00011+-0.00028 (0.743) | +0.00008+-0.00033 | +0.00009+-0.00029 | +0.00020 +- 0.00021 | 0.109 |
| I(CLq) | 4 | 2401 | +0.00070 | +0.00048 +- 0.00028 (+1.7) | 0.040 | +0.00022+-0.00023 | +0.00030+-0.00021 | 0.085 | 0.075 | +0.00039+-0.00033 | -0.00009+-0.00026 (0.752) | -0.00027+-0.00020 | +0.00010+-0.00027 | +0.00019 +- 0.00021 | 0.099 |
| J(CLq,Hq) | 4 | 1764 | +0.00076 | +0.00048 +- 0.00030 (+1.6) | 0.025 | +0.00021+-0.00024 | +0.00029+-0.00021 | 0.075 | 0.085 | +0.00019+-0.00033 | +0.00014+-0.00028 (0.386) | -0.00033+-0.00028 | +0.00015+-0.00029 | +0.00031 +- 0.00022 | 0.059 |
| K(Mq,Hq) | 4 | 2401 | +0.00083 | +0.00047 +- 0.00026 (+1.8) | 0.015 | +0.00021+-0.00024 | +0.00029+-0.00021 | 0.085 | 0.065 | +0.00028+-0.00028 | +0.00018+-0.00024 (0.347) | +0.00038+-0.00023 | +0.00030+-0.00023 | +0.00032 +- 0.00020 | 0.030 |
| J(Mq,Hq) | 4 | 1764 | +0.00079 | +0.00047 +- 0.00032 (+1.5) | 0.030 | +0.00021+-0.00026 | +0.00029+-0.00020 | 0.030 | 0.085 | +0.00068+-0.00030 | -0.00013+-0.00026 (0.802) | +0.00028+-0.00028 | +0.00026+-0.00026 | +0.00017 +- 0.00023 | 0.198 |
| C(M50) | 2 | 49 | +0.00053 | +0.00018 +- 0.00030 (+0.6) | 0.189 | -0.00008+-0.00020 | 0 | - | 0.796 | +0.00021+-0.00025 | +0.00055+-0.00026 (0.040) | +0.00028+-0.00029 | +0.00027+-0.00026 | +0.00036 +- 0.00022 | 0.040 |
| C(Mq) | 2 | 49 | +0.00052 | +0.00023 +- 0.00030 (+0.8) | 0.144 | -0.00003+-0.00022 | +0.00005+-0.00011 | 0.299 | 0.627 | +0.00007+-0.00027 | +0.00026+-0.00023 (0.218) | +0.00044+-0.00021 | +0.00024+-0.00022 | +0.00025 +- 0.00021 | 0.069 |
| G(M3q) | 3 | 343 | +0.00070 | +0.00031 +- 0.00029 (+1.1) | 0.075 | +0.00005+-0.00021 | +0.00013+-0.00014 | 0.159 | 0.179 | +0.00010+-0.00025 | +0.00044+-0.00027 (0.079) | +0.00028+-0.00028 | +0.00039+-0.00028 | +0.00037 +- 0.00023 | 0.020 |
| G(M3r) | 3 | 343 | +0.00067 | +0.00033 +- 0.00028 (+1.2) | 0.055 | +0.00007+-0.00020 | +0.00015+-0.00019 | 0.134 | 0.189 | +0.00012+-0.00027 | +0.00006+-0.00024 (0.436) | +0.00008+-0.00021 | +0.00028+-0.00024 | +0.00020 +- 0.00021 | 0.149 |
| D(M50) | 3 | 294 | +0.00054 | -0.00001 +- 0.00028 (-0.0) | 0.453 | -0.00027+-0.00021 | -0.00019+-0.00013 | 0.905 | 0.866 | +0.00040+-0.00032 | +0.00025+-0.00027 (0.188) | +0.00010+-0.00030 | +0.00016+-0.00027 | +0.00012 +- 0.00021 | 0.307 |
| I(M50) | 4 | 2401 | +0.00059 | +0.00007 +- 0.00027 (+0.3) | 0.388 | -0.00019+-0.00019 | -0.00011+-0.00009 | 0.771 | 0.662 | +0.00014+-0.00030 | +0.00049+-0.00026 (0.040) | +0.00031+-0.00028 | +0.00022+-0.00026 | +0.00028 +- 0.00021 | 0.079 |
| A | 2 | 42 | +0.00054 | +0.00027 +- 0.00031 (+0.9) | 0.114 | +0.00000+-0.00026 | +0.00009+-0.00021 | 0.244 | - | +0.00031+-0.00032 | +0.00006+-0.00029 (0.525) | -0.00026+-0.00018 | +0.00015+-0.00031 | +0.00016 +- 0.00023 | 0.178 |
| B | 3 | 294 | +0.00054 | +0.00025 +- 0.00030 (+0.8) | 0.119 | -0.00001+-0.00025 | +0.00007+-0.00021 | 0.294 | - | +0.00024+-0.00029 | +0.00001+-0.00030 (0.594) | -0.00019+-0.00021 | -0.00008+-0.00032 | +0.00013 +- 0.00024 | 0.178 |
| E(CS45) | 2 | 36 | +0.00002 | -0.00026 +- 0.00007 (-3.5) | 1.000 | -0.00052+-0.00021 | -0.00044+-0.00029 | 0.970 | 0.746 | -0.00016+-0.00012 | -0.00023+-0.00016 (1.000) | +0.00019+-0.00017 | -0.00018+-0.00015 | -0.00025 +- 0.00009 | 1.000 |
| E(CLq) | 2 | 36 | +0.00010 | -0.00011 +- 0.00012 (-0.9) | 0.900 | -0.00037+-0.00023 | -0.00029+-0.00029 | 0.866 | 0.209 | +0.00005+-0.00016 | -0.00026+-0.00015 (1.000) | -0.00003+-0.00017 | -0.00017+-0.00016 | -0.00019 +- 0.00009 | 1.000 |
| E(Hq) | 2 | 36 | +0.00006 | -0.00030 +- 0.00018 (-1.6) | 1.000 | -0.00056+-0.00026 | -0.00048+-0.00031 | 0.970 | 0.652 | -0.00010+-0.00018 | -0.00043+-0.00018 (1.000) | -0.00012+-0.00018 | -0.00017+-0.00018 | -0.00037 +- 0.00014 | 1.000 |
| L(CL3q) | 3 | 216 | +0.00011 | -0.00036 +- 0.00011 (-3.2) | 1.000 | -0.00062+-0.00022 | -0.00054+-0.00029 | 0.995 | 0.766 | -0.00014+-0.00014 | +0.00018+-0.00017 (0.069) | +0.00010+-0.00018 | +0.00005+-0.00017 | -0.00009 +- 0.00010 | 0.931 |
| C(DAY) [diag] | 2 | 49 | +0.00048 | +0.00027 +- 0.00028 (+1.0) | 0.164 | +0.00001+-0.00021 | +0.00009+-0.00018 | 0.303 | 0.478 | +0.00038+-0.00029 | -0.00012+-0.00030 (0.752) | -0.00041+-0.00019 | +0.00013+-0.00031 | +0.00007 +- 0.00023 | 0.277 |

Selection variants (family chosen inside the training part):

| variant | C1 nested - plain | p_A | C1 - fixed M-rule | C2 | C3 | C4 | families chosen most often on C1 (of 25 fits) |
|---|---|---|---|---|---|---|---|
| B-all | +0.00001+-0.00025 | 0.448 | -0.00026+-0.00020 | +0.00059+-0.00029 | -0.00000+-0.00024 | +0.00032+-0.00025 | F(Hq) x5, J(U0,Hq) x3, K(Mq,Hq) x2, J(Hq,Hq) x2 |
| B-3 | +0.00016+-0.00027 | 0.174 | -0.00010+-0.00020 | +0.00035+-0.00027 | +0.00008+-0.00023 | -0.00012+-0.00022 | F(Hq) x10, G(M3q) x3, G(M3r) x3, G(H3q) x2 |
| G | +0.00004+-0.00025 | 0.413 | -0.00022+-0.00021 | +0.00061+-0.00030 | +0.00036+-0.00026 | +0.00025+-0.00026 | K(Mq,Hq) x10, J(Hq,Hq) x4, J(Mq,Hq) x3, H(M3r) x2 |

The fixed M-rule and the M-rule family:

| fixed M-rule (0.2 / 0.4 by M >= 50, w_ft 0.5) minus plain | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| val | +0.00026+-0.00021 | +0.00051+-0.00021 | +0.00071+-0.00021 | +0.00043+-0.00019 |
| pmix public | +0.00027+-0.00029 | +0.00059+-0.00028 | +0.00081+-0.00027 | +0.00076+-0.00026 |
| pmix private | +0.00038+-0.00036 | +0.00072+-0.00034 | +0.00093+-0.00034 | +0.00092+-0.00033 |
| fixed M-rule minus nested C(M50) | +0.00008+-0.00020 | +0.00030+-0.00014 | +0.00016+-0.00015 | +0.00015+-0.00020 |
| nested C(M50) minus plain, pmix public | +0.00027+-0.00039 | +0.00043+-0.00032 | +0.00070+-0.00034 | +0.00057+-0.00037 |
| nested C(M50) minus plain, pmix private | +0.00040+-0.00049 | +0.00062+-0.00039 | +0.00090+-0.00043 | +0.00070+-0.00046 |

Rules the M-rule family picks in-sample (w_pop on M < 50 | M >= 50): C1: a 0.4 | a 0.5 ; C2: a 0.2 | a 0.4 ; C3: a 0.2 | a 0.5 ; C4: a 0.1 | a 0.5. Constant family A (w_pop, w_pop, w_ft): C1: (0.4, 0.4, 0.7) ; C2: (0.4, 0.4, 0.7) ; C3: (0.4, 0.4, 0.4) ; C4: (0.2, 0.2, 0.6). Three-step G(M3r): C1: a 0.1 | a 0.4 | a 0.6 ; C2: a 0.2 | a 0.4 | a 0.4 ; C3: a 0.2 | a 0.3 | a 0.5 ; C4: a 0.1 | a 0.2 | a 0.4. The fitted values move from member set to member set; only "more on long feeds" is common.

Eight-member popularity-light pools (addendum 6, declared after the h2 re-read; C5 = FD16, C6 = FD8, C7 = W16, C8 = W8, each + 4 wide + 4 d128; same search, same folds):

```
=== C5 FD16 + 4 wide + 4 d128: plain 0.40272 | fixed M-rule - plain -0.00010+-0.00020 | top nested row I(CS45) | 6s
family         | in-sample | nested - plain (z)          | nested - fixed MRULE | best rule on all val
C(M50)         | +0.00019  | -0.00011+-0.00027 (-0.4) | -0.00001+-0.00022   | (0.1, 0.1, 0.5) ; (0.6, 0.6, 0.5)
G(M3q)         | +0.00035  | +0.00008+-0.00028 (+0.3) | +0.00018+-0.00024   | (0.1, 0.1, 0.5) ; (0.5, 0.5, 0.5) ; (0.2, 0.2, 0.5)
K(Mq,Hq)       | +0.00048  | +0.00008+-0.00025 (+0.3) | +0.00019+-0.00025   | (0.5, 0.5, 0.5) ; (0.1, 0.1, 0.5) ; (0.6, 0.6, 0.5) ; (0.2, 0.2, 0.5)
F(Hq)          | +0.00007  | -0.00054+-0.00022 (-2.4) | -0.00043+-0.00022   | (0.4, 0.4, 0.7) ; (0.4, 0.4, 0.6)
A              | +0.00001  | -0.00031+-0.00020 (-1.5) | -0.00021+-0.00020   | (0.4, 0.4, 0.6)
I(M50)         | +0.00056  | +0.00024+-0.00022 (+1.1) | +0.00035+-0.00017   | (0.1, 0.2, 0.5) ; (0.4, 0.5, 0.5)
VARIANT B-3: nested - plain -0.00041+-0.00019 | - fixed MRULE -0.00031+-0.00019
=== C6 FD8 + 4 wide + 4 d128: plain 0.40238 | fixed M-rule - plain +0.00047+-0.00020 | top nested row J(Mq,U0) | 7s
family         | in-sample | nested - plain (z)          | nested - fixed MRULE | best rule on all val
C(M50)         | +0.00047  | +0.00028+-0.00022 (+1.3) | -0.00019+-0.00007   | (0.2, 0.2, 0.5) ; (0.4, 0.4, 0.5)
G(M3q)         | +0.00055  | +0.00018+-0.00022 (+0.8) | -0.00029+-0.00013   | (0.2, 0.2, 0.5) ; (0.4, 0.4, 0.5) ; (0.6, 0.6, 0.5)
K(Mq,Hq)       | +0.00062  | +0.00010+-0.00028 (+0.4) | -0.00037+-0.00023   | (0.4, 0.4, 0.5) ; (0.1, 0.1, 0.5) ; (0.6, 0.6, 0.5) ; (0.4, 0.4, 0.5)
F(Hq)          | +0.00048  | +0.00004+-0.00029 (+0.1) | -0.00043+-0.00026   | (0.5, 0.5, 0.6) ; (0.5, 0.5, 0.4)
A              | +0.00039  | -0.00005+-0.00028 (-0.2) | -0.00052+-0.00025   | (0.3, 0.3, 0.7)
I(M50)         | +0.00059  | +0.00023+-0.00026 (+0.9) | -0.00024+-0.00017   | (0.3, 0.2, 0.5) ; (0.6, 0.3, 0.5)
VARIANT B-3: nested - plain +0.00042+-0.00028 | - fixed MRULE -0.00005+-0.00021
=== C7 W16 + 4 wide + 4 d128: plain 0.40189 | fixed M-rule - plain +0.00064+-0.00021 | top nested row K(M50,CS45) | 8s
family         | in-sample | nested - plain (z)          | nested - fixed MRULE | best rule on all val
C(M50)         | +0.00092  | +0.00074+-0.00027 (+2.7) | +0.00010+-0.00018   | (0.1, 0.1, 0.5) ; (0.4, 0.4, 0.5)
G(M3q)         | +0.00076  | +0.00029+-0.00026 (+1.1) | -0.00035+-0.00017   | (0.1, 0.1, 0.5) ; (0.4, 0.4, 0.5) ; (0.4, 0.4, 0.5)
K(Mq,Hq)       | +0.00067  | +0.00002+-0.00026 (+0.1) | -0.00061+-0.00021   | (0.4, 0.4, 0.5) ; (0.5, 0.5, 0.5) ; (0.4, 0.4, 0.5) ; (0.3, 0.3, 0.5)
F(Hq)          | +0.00066  | -0.00002+-0.00033 (-0.1) | -0.00065+-0.00027   | (0.4, 0.4, 0.5) ; (0.4, 0.4, 0.4)
A              | +0.00055  | -0.00007+-0.00031 (-0.2) | -0.00070+-0.00025   | (0.4, 0.4, 0.5)
I(M50)         | +0.00098  | +0.00057+-0.00024 (+2.4) | -0.00006+-0.00016   | (0.2, 0.0, 0.5) ; (0.4, 0.4, 0.5)
VARIANT B-3: nested - plain +0.00016+-0.00025 | - fixed MRULE -0.00047+-0.00016
=== C8 W8 + 4 wide + 4 d128: plain 0.40224 | fixed M-rule - plain +0.00037+-0.00021 | top nested row K(M50,Hq) | 9s
family         | in-sample | nested - plain (z)          | nested - fixed MRULE | best rule on all val
C(M50)         | +0.00037  | +0.00009+-0.00022 (+0.4) | -0.00028+-0.00010   | (0.2, 0.2, 0.5) ; (0.4, 0.4, 0.5)
G(M3q)         | +0.00029  | -0.00014+-0.00023 (-0.6) | -0.00051+-0.00018   | (0.1, 0.1, 0.5) ; (0.4, 0.4, 0.5) ; (0.4, 0.4, 0.5)
K(Mq,Hq)       | +0.00044  | -0.00008+-0.00022 (-0.3) | -0.00045+-0.00021   | (0.2, 0.2, 0.5) ; (0.3, 0.3, 0.5) ; (0.6, 0.6, 0.5) ; (0.1, 0.1, 0.5)
F(Hq)          | +0.00026  | -0.00021+-0.00023 (-0.9) | -0.00058+-0.00025   | (0.2, 0.2, 0.5) ; (0.2, 0.2, 0.3)
A              | +0.00024  | -0.00007+-0.00032 (-0.2) | -0.00044+-0.00029   | (0.4, 0.4, 0.3)
I(M50)         | +0.00050  | -0.00011+-0.00024 (-0.5) | -0.00048+-0.00015   | (0.2, 0.0, 0.5) ; (0.3, 0.5, 0.5)
VARIANT B-3: nested - plain -0.00023+-0.00020 | - fixed MRULE -0.00061+-0.00020
```

The M-rule family is positive on 3 of these 4 (not on FD16 + 8); F(Hq) reads -0.00054 / +0.00004 / -0.00002 / -0.00021 (the first on the FD16 base where it had topped the table) and averages -0.00001 over the 8 compositions. Summary over all 8 (shared feeds and members; the ranking is post hoc):

```
fixed MRULE minus PLAIN on C1..C8: +0.00026 +0.00051 +0.00071 +0.00043 -0.00010 +0.00047 +0.00064 +0.00037 | mean +0.00041 | positive 7 of 8
  per-feed mean over the 8 compositions (shared feeds and members; paired SE): +0.00041+-0.00014
nested C(M50) minus PLAIN on C1..C8: +0.00018 +0.00021 +0.00055 +0.00028 -0.00011 +0.00028 +0.00074 +0.00009 | mean +0.00028 | positive 7 of 8 | per-feed mean over the 8, paired SE: +0.00028+-0.00017
eligible families positive (nested > PLAIN) on all 8: 0 : 
eligible families positive on >= 7 of 8: 11 : C(M50) (+0.00028), I(M50) (+0.00024), I(Mq) (+0.00012), I(Hq) (+0.00003), G(M3q) (+0.00019), G(H3q) (+0.00022), K(M50,CS45) (+0.00026), K(M50,CLq) (+0.00025), K(M50,Hq) (+0.00035), K(Mq,CS45) (+0.00009), K(Mq,Hq) (+0.00018)
top 8 eligible families by the mean nested gain over the 8 compositions (post hoc ranking, selection bias applies):
  K(M50,Hq)      mean +0.00035 | positive 7 of 8 | -0.00004 +0.00046 +0.00061 +0.00059 +0.00010 +0.00023 +0.00045 +0.00042 | minus fixed MRULE (per-feed mean over 8): -0.00006+-0.00011
  C(M50)         mean +0.00028 | positive 7 of 8 | +0.00018 +0.00021 +0.00055 +0.00028 -0.00011 +0.00028 +0.00074 +0.00009 | minus fixed MRULE (per-feed mean over 8): -0.00013+-0.00007
  K(M50,CS45)    mean +0.00026 | positive 7 of 8 | +0.00008 +0.00018 +0.00037 +0.00034 -0.00014 +0.00049 +0.00077 +0.00001 | minus fixed MRULE (per-feed mean over 8): -0.00015+-0.00008
  K(M50,CLq)     mean +0.00025 | positive 7 of 8 | +0.00001 +0.00039 +0.00049 +0.00035 -0.00013 +0.00026 +0.00049 +0.00014 | minus fixed MRULE (per-feed mean over 8): -0.00016+-0.00008
  I(M50)         mean +0.00024 | positive 7 of 8 | +0.00007 +0.00014 +0.00049 +0.00031 +0.00024 +0.00023 +0.00057 -0.00011 | minus fixed MRULE (per-feed mean over 8): -0.00017+-0.00008
  K(M50,U0)      mean +0.00023 | positive 5 of 8 | +0.00025 -0.00005 +0.00043 +0.00024 -0.00006 +0.00032 +0.00070 -0.00002 | minus fixed MRULE (per-feed mean over 8): -0.00018+-0.00009
  G(H3q)         mean +0.00022 | positive 7 of 8 | +0.00027 +0.00059 +0.00008 -0.00011 +0.00015 +0.00024 +0.00046 +0.00012 | minus fixed MRULE (per-feed mean over 8): -0.00018+-0.00016
  I(CS45)        mean +0.00022 | positive 6 of 8 | +0.00030 +0.00063 +0.00048 -0.00014 +0.00034 +0.00002 +0.00030 -0.00014 | minus fixed MRULE (per-feed mean over 8): -0.00019+-0.00013
F(Hq): +0.00076 +0.00041 -0.00019 -0.00034 -0.00054 +0.00004 -0.00002 -0.00021 | mean -0.00001
families above the fixed MRULE on all 8: 0; above nested C(M50) on all 8: 0
```

No eligible family is positive on all 8; none is above the fixed M-rule on all 8; none is above the M-rule family on all 8. The five best families by the 8-composition mean all gate w_pop on M >= 50, and each is below the fixed M-rule (best: K(M50,Hq) +0.00035, -0.00006 +- 0.00011 against the fixed rule).

Declared criteria, all eligible families (top rows by C1 nested):

```
eligible families 95 | pass c1: 3 | pass c2: 5 | pass c3: 4 | pass c4: 45 | pass c5: 68 | c1 & c2: 3 | c1 & c2 & c3: 0 | ALL: 0
families with nested > PLAIN on all 4 compositions: 22 of 95; with nested > nested C(M50) on all 4: 0; with nested > fixed MRULE on all 4: 0
family           par | C1 nested-PLAIN (z)        p_A   | - nested C(M50)  p_A,M | - fixed MRULE | C2 / C3 / C4 nested - nested C(M50) | all4>PLAIN | pmix-priv vs MRULE | pmix-pub vs PLAIN | min test cell | c1 c2 c3 c4 c5
F(Hq)            3   | +0.00076+-0.00032 (+2.4) 0.010 | +0.00058  0.005       | +0.00050      | +0.00020 / -0.00074 / -0.00062      | False      | +0.00091           | +0.00108          | 0.498         | Y  Y  n  Y  Y 
J(U0,Hq)         4   | +0.00072+-0.00031 (+2.3) 0.010 | +0.00054  0.005       | +0.00045      | +0.00011 / -0.00068 / -0.00057      | False      | +0.00076           | +0.00096          | 0.001         | Y  Y  n  Y  n 
J(CS45,Hq)       4   | +0.00066+-0.00032 (+2.0) 0.010 | +0.00047  0.005       | +0.00039      | +0.00003 / -0.00034 / -0.00064      | False      | +0.00057           | +0.00084          | 0.103         | Y  Y  n  Y  Y 
J(Hq,Hq)         4   | +0.00051+-0.00028 (+1.8) 0.030 | +0.00033  0.050       | +0.00025      | +0.00006 / -0.00066 / -0.00020      | False      | +0.00075           | +0.00086          | 0.498         | n  Y  n  Y  Y 
I(CLq)           4   | +0.00048+-0.00028 (+1.7) 0.040 | +0.00030  0.085       | +0.00022      | +0.00018 / -0.00064 / -0.00055      | False      | +0.00038           | +0.00062          | 0.384         | n  n  n  Y  Y 
J(CLq,Hq)        4   | +0.00048+-0.00030 (+1.6) 0.025 | +0.00029  0.075       | +0.00021      | -0.00002 / -0.00040 / -0.00060      | False      | +0.00039           | +0.00064          | 0.190         | n  n  n  Y  Y 
K(Mq,Hq)         4   | +0.00047+-0.00026 (+1.8) 0.015 | +0.00029  0.085       | +0.00021      | +0.00007 / -0.00037 / +0.00010      | True       | +0.00013           | +0.00050          | 0.240         | n  n  Y  Y  Y 
J(Mq,Hq)         4   | +0.00047+-0.00032 (+1.5) 0.030 | +0.00029  0.030       | +0.00021      | +0.00047 / -0.00067 / -0.00000      | False      | +0.00058           | +0.00074          | 0.240         | n  Y  n  Y  Y 
K(CS45,U0)       4   | +0.00046+-0.00032 (+1.4) 0.025 | +0.00027  0.060       | +0.00019      | +0.00017 / -0.00049 / -0.00054      | False      | +0.00012           | +0.00040          | 0.018         | n  n  n  Y  n 
C(U0)            2   | +0.00044+-0.00031 (+1.4) 0.040 | +0.00026  0.055       | +0.00018      | +0.00011 / -0.00047 / -0.00056      | False      | +0.00024           | +0.00046          | 0.041         | n  n  n  Y  n 
```
```
FALLBACK (block 6): C(M50) nested - PLAIN on C1 +0.00018+-0.00030 (z +0.61), p_A 0.189; on C2 / C3 / C4: +0.00021+-0.00025 / +0.00055+-0.00026 / +0.00028+-0.00029 -> "is not shown (primary composition); positive on all 4"
```

Reading. (a) c1 + c2 are met only by F(Hq), J(U0,Hq), J(CS45,Hq): one finding (w_pop 0.4 everywhere + w_ft 0.7 on H < 190), three times. Its C1 gain is +0.00048 from "0.4 everywhere" (which reads +0.00022 / -0.00007 on the twins) plus +0.00028 from the w_ft step (which the twins fit with the opposite sign). (b) Every w_ft-gated family at w_pop 0.2 (E, L) is at or below zero nested. (c) Per-branch weights (B, I) add nothing. (d) The best-family variants land at +0.00001 (B-all) / +0.00016 (B-3) on C1, below the fixed M-rule.

## 4. Placebo calibration (step 4) [M]

PL-A on the FD16 members (observed = C1):

```
################ observed = C1 ('FD16', 'FDP') | PL-A on FD16: 200 draws | PL-G on C1: 200 permutations
PL-A whole-search calibration (gains over the draw's own PLAIN):
  nested, all 95 eligible families pooled: mean -0.00002 (SE of the mean over draws 0.00001); per-family sd over draws: median 0.00022
  in-sample, per family: mean over families of the mean optimism (in-sample minus nested) +0.00035; by number of parameters: 2 par +0.00021, 3 par +0.00032, 4 par +0.00039
  greedy in-sample best of all eligible rules (the "scan"): mean +0.00057 sd 0.00014 q05 +0.00034 q50 +0.00056 q95 +0.00082 | observed +0.00083 (p 0.055)
  best NESTED family per draw (max over 95 families; the bias of picking the top row of a nested table): mean +0.00038 q95 +0.00068 | observed top row +0.00076 (F(Hq)) p 0.030
  variant B-all : placebo nested mean -0.00001 sd 0.00019 q95 +0.00034 | observed +0.00001 p_A 0.448
  variant B-3   : placebo nested mean -0.00002 sd 0.00020 q95 +0.00040 | observed +0.00016 p_A 0.174
  variant G     : placebo nested mean +0.00001 sd 0.00023 q95 +0.00040 | observed +0.00004 p_A 0.413
  fixed rule 0.2 / 0.4 by M (not searched) on placebo pools: mean +0.00004 sd 0.00015 q95 +0.00026 | observed +0.00026 p 0.055
```

PL-A on the W16 members (observed = C3):

```
################ observed = C3 ('W16', 'P1w') | PL-A on W16: 100 draws
PL-A whole-search calibration (gains over the draw's own PLAIN):
  nested, all 95 eligible families pooled: mean +0.00006 (SE of the mean over draws 0.00002); per-family sd over draws: median 0.00024
  in-sample, per family: mean over families of the mean optimism (in-sample minus nested) +0.00034; by number of parameters: 2 par +0.00020, 3 par +0.00030, 4 par +0.00038
  greedy in-sample best of all eligible rules (the "scan"): mean +0.00063 sd 0.00015 q05 +0.00042 q50 +0.00061 q95 +0.00091 | observed +0.00099 (p 0.030)
  best NESTED family per draw (max over 95 families; the bias of picking the top row of a nested table): mean +0.00044 q95 +0.00076 | observed top row +0.00061 (K(M50,Hq)) p 0.198
  variant B-all : placebo nested mean +0.00005 sd 0.00021 q95 +0.00044 | observed -0.00000 p_A 0.554
  variant B-3   : placebo nested mean +0.00006 sd 0.00022 q95 +0.00049 | observed +0.00008 p_A 0.396
  variant G     : placebo nested mean +0.00008 sd 0.00024 q95 +0.00050 | observed +0.00036 p_A 0.119
  fixed rule 0.2 / 0.4 by M (not searched) on placebo pools: mean +0.00006 sd 0.00008 q95 +0.00020 | observed +0.00071 p 0.010
```

- The nested procedure does not manufacture gains: pooled placebo nested mean -0.00002 (FD16), +0.00006 (W16). The W16 value tripped the stop rule I had declared ("positive beyond 2 SE of the mean over draws"); that rule was badly specified (all draws share the same 14,423 feeds, so the mean over draws keeps the feed noise). Leak check declared (addendum 2) and run: label-permutation null of the whole search, 300 independent permutations, pooled nested gain **-0.000006 +- 0.000010** (variants z -0.5 .. +1.1): no leak. The W16 placebo mean is explained by the grid point 0.2 being slightly low for those pools (constant 0.1 and 0.4 read +0.00007 in 70-80% of the draws).

```
L1 label-permutation null, 60 permutations: pooled nested gain mean +0.000036 SE 0.000025 (z +1.44) | per-family sd 0.00026 | in-sample optimism mean +0.00041
  variant B-all: mean +0.000038 SE 0.000030 (z +1.28)
  variant B-3: mean +0.000030 SE 0.000029 (z +1.06)
  variant G: mean +0.000041 SE 0.000034 (z +1.22)
  per-family z of the mean: min -0.40 max +3.31 mean +1.16; share |z| > 2: 0.126
L1 PASS (no leak)
```
```
L1 label-permutation null, 240 permutations: pooled nested gain mean -0.000017 SE 0.000010 (z -1.67) | per-family sd 0.00024 | in-sample optimism mean +0.00036
  variant B-all: mean -0.000002 SE 0.000014 (z -0.14)
  variant B-3: mean -0.000015 SE 0.000013 (z -1.16)
  variant G: mean +0.000011 SE 0.000017 (z +0.65)
  per-family z of the mean: min -2.46 max +0.44 mean -1.10; share |z| > 2: 0.074
L1 PASS (no leak)
```
```
L2 W16 placebo, 40 draws: const a (c = .5) minus PLAIN, mean over draws [share of draws > 0]: a 0.0: +0.00004 [0.68] | a 0.1: +0.00007 [0.80] | a 0.2: +0.00000 [0.00] | a 0.3: +0.00002 [0.60] | a 0.4: +0.00007 [0.70] | a 0.5: +0.00004 [0.57] | a 0.6: -0.00001 [0.38]
L2 W16 placebo: const c (a = .2) minus PLAIN: c 0.3: +0.00001 [0.53] | c 0.4: -0.00000 [0.50] | c 0.5: +0.00000 [0.00] | c 0.6: -0.00001 [0.42] | c 0.7: +0.00000 [0.55] | c 0.8: +0.00002 [0.55]
L2 FD16 placebo, 40 draws: const a (c = .5) minus PLAIN, mean over draws [share of draws > 0]: a 0.0: -0.00001 [0.45] | a 0.1: +0.00003 [0.55] | a 0.2: +0.00000 [0.00] | a 0.3: +0.00001 [0.55] | a 0.4: +0.00006 [0.62] | a 0.5: -0.00002 [0.50] | a 0.6: -0.00009 [0.38]
L2 FD16 placebo: const c (a = .2) minus PLAIN: c 0.3: +0.00000 [0.55] | c 0.4: -0.00001 [0.40] | c 0.5: +0.00000 [0.00] | c 0.6: -0.00001 [0.45] | c 0.7: -0.00001 [0.53] | c 0.8: -0.00001 [0.40]
```

- Noise of one nested number across member draws: sd 0.00022 (single member set), 0.00016 (mean of two member sets). Correlation of the FD16 and W16 placebo gains over matched draws: -0.01, i.e. the nested noise is member-combination noise, not shared feed noise.
- In-sample optimism of a fitted rule: +0.00021 (2 parameters), +0.00032 (3), +0.00039 (4); greedy scan over all rules +0.00057 (q95 +0.00082) on FD16, +0.00063 (q95 +0.00091) on W16; under pure label noise +0.00062 (q95 +0.00088).
- The fixed 0.2 / 0.4 rule applied to placebo pools reads +0.00004 (sd 0.00015) on FD16 and +0.00006 (sd 0.00008) on W16: observed +0.00026 (p 0.055) on C1, +0.00071 (p 0.010, above all 100 draws) on C3. The rule was chosen on these feeds, so these p-values are optimistic; they do show that the effect belongs to the popularity-light pool and not to "any 4-member pool".

Two-member-set statistic with the matched placebo (addendum 3; post hoc in design):

```
PL-A2: 100 matched draws. placebo T2 pooled over eligible families: mean +0.00000; per-family sd over draws: median 0.00016 (single member set: 0.00022 FD16, 0.00022 W16)
correlation over draws of the FD16 and W16 placebo nested gains (per family, median): -0.01
C(M50): T2 +0.00036+-0.00022 (paired SE of the two-set mean; z +1.62) | placebo mean +0.00002 sd 0.00016 q95 +0.00030 | p2 0.040 | pmix-public +0.00049+-0.00030 private +0.00065+-0.00039
fixed MRULE: T2 +0.00048+-0.00017 (z +2.86) | the same fixed rule on the placebo pools: mean +0.00004 sd 0.00010 max +0.00027 | p2 0.010 (the rule was chosen on these feeds: optimistic) | pmix-public +0.00054+-0.00023 private +0.00065+-0.00029
family-wise: max over 95 families of placebo T2 per draw: mean +0.00027 q95 +0.00050 | observed max +0.00043 (J(CS45,Hq)) p 0.129
family-wise for "beats C(M50)": max over families of placebo [T2(fam) - T2(C(M50))] per draw: mean +0.00025 q95 +0.00046 | observed max +0.00007 (J(CS45,Hq)) p 0.941
variant B-all: T2 +0.00000+-0.00020 | placebo mean +0.00000 sd 0.00014 | p2 0.475
variant B-3: T2 +0.00012+-0.00020 | placebo mean +0.00001 sd 0.00014 | p2 0.248
variant G: T2 +0.00020+-0.00020 | placebo mean +0.00002 sd 0.00015 | p2 0.139
family           par | T2 (paired SE)        z   | C1       C3      | placebo T2: mean    sd      q95     p2   | T2 - T2 C(M50)  placebo diff q95  p2,M | T2 - fixed MRULE
```

Cross-model read (addendum 1; post hoc in design). "xm-same" = rule fitted on all val feeds of the other member set (other models, same feeds: the kind of evidence on record for the M-rule); "xm-nest" = other models and other users.

```
=== SRC C3 -> TGT C1: fixed MRULE - PLAIN on TGT +0.00026+-0.00021 | eligible 2-3 par families: mean own in-sample +0.00043 xm-same +0.00018 xm-nest +0.00007 own nested +0.00009 pool-nest +0.00009 | shared-feed share = 1 - xm-nest / xm-same = 0.62
F(Hq)            3   | +0.00076      | +0.00016 | -0.00005+-0.00029    | +0.00076   | +0.00040+-0.00030      | +0.00014+-0.00024
K(Mq,Hq)         4   | +0.00083      | +0.00051 | +0.00030+-0.00023    | +0.00047   | +0.00054+-0.00028      | +0.00028+-0.00025
C(M50)           2   | +0.00053      | +0.00032 | +0.00027+-0.00026    | +0.00018   | +0.00029+-0.00028      | +0.00002+-0.00017
G(M3q)           3   | +0.00070      | +0.00034 | +0.00039+-0.00028    | +0.00031   | +0.00042+-0.00028      | +0.00015+-0.00018
A                2   | +0.00054      | +0.00032 | +0.00015+-0.00031    | +0.00027   | +0.00031+-0.00031      | +0.00005+-0.00025
=== SRC C1 -> TGT C3: fixed MRULE - PLAIN on TGT +0.00071+-0.00021 | eligible 2-3 par families: mean own in-sample +0.00041 xm-same +0.00004 xm-nest +0.00004 own nested -0.00003 pool-nest +0.00008 | shared-feed share = 1 - xm-nest / xm-same = -0.02
C(M50)           2   | +0.00080      | +0.00031 | +0.00036+-0.00028    | +0.00055   | +0.00048+-0.00027      | -0.00023+-0.00017
G(M3q)           3   | +0.00071      | +0.00053 | +0.00034+-0.00028    | +0.00044   | +0.00035+-0.00027      | -0.00035+-0.00017
A                2   | +0.00044      | -0.00015 | -0.00007+-0.00030    | +0.00006   | +0.00013+-0.00030      | -0.00057+-0.00024
F(Hq)            3   | +0.00047      | +0.00013 | +0.00013+-0.00031    | -0.00019   | +0.00013+-0.00029      | -0.00058+-0.00024
K(Mq,Hq)         4   | +0.00070      | +0.00045 | +0.00024+-0.00025    | +0.00018   | +0.00041+-0.00028      | -0.00030+-0.00024
=== SRC C4 -> TGT C2: fixed MRULE - PLAIN on TGT +0.00051+-0.00021 | eligible 2-3 par families: mean own in-sample +0.00052 xm-same +0.00012 xm-nest -0.00000 own nested +0.00015 pool-nest +0.00015 | shared-feed share = 1 - xm-nest / xm-same = 1.03
C(M50)           2   | +0.00051      | +0.00017 | +0.00014+-0.00029    | +0.00021   | +0.00025+-0.00023      | -0.00026+-0.00008
G(M3q)           3   | +0.00056      | +0.00034 | +0.00020+-0.00029    | +0.00010   | +0.00029+-0.00026      | -0.00022+-0.00015
A                2   | +0.00065      | -0.00010 | -0.00024+-0.00017    | +0.00031   | +0.00037+-0.00025      | -0.00013+-0.00023
F(Hq)            3   | +0.00078      | +0.00042 | +0.00014+-0.00023    | +0.00041   | +0.00033+-0.00025      | -0.00017+-0.00024
K(Mq,Hq)         4   | +0.00071      | +0.00047 | +0.00027+-0.00025    | +0.00028   | +0.00023+-0.00024      | -0.00028+-0.00019
=== SRC C2 -> TGT C4: fixed MRULE - PLAIN on TGT +0.00043+-0.00019 | eligible 2-3 par families: mean own in-sample +0.00032 xm-same -0.00013 xm-nest -0.00016 own nested -0.00008 pool-nest -0.00007 | shared-feed share = 1 - xm-nest / xm-same = -0.21
K(Mq,Hq)         4   | +0.00076      | +0.00038 | +0.00027+-0.00025    | +0.00038   | +0.00038+-0.00022      | -0.00005+-0.00017
C(M50)           2   | +0.00056      | +0.00043 | +0.00008+-0.00023    | +0.00028   | +0.00029+-0.00021      | -0.00014+-0.00009
G(M3q)           3   | +0.00056      | +0.00028 | +0.00016+-0.00023    | +0.00028   | +0.00023+-0.00025      | -0.00020+-0.00015
A                2   | +0.00020      | -0.00039 | -0.00047+-0.00030    | -0.00026   | -0.00015+-0.00023      | -0.00057+-0.00022
F(Hq)            3   | +0.00025      | -0.00044 | -0.00029+-0.00030    | -0.00034   | +0.00001+-0.00024      | -0.00041+-0.00021
```

An average 2-3 parameter family keeps +0.00018 / +0.00004 / +0.00012 / -0.00013 of its +0.0003..+0.0005 in-sample gain when the member set changes on the SAME feeds: most of an in-sample gain is specific to the member combination. The M-rule family is the exception that carries: +0.00027 / +0.00036 / +0.00014 / +0.00008 with other models and other users. "Six compositions all positive" on record is therefore weaker than six confirmations (shared feeds, shared members), but it is not empty.

## 5. Declared single hypotheses h1-h4 (step 6; fixed rules, no search) [M]

| hypothesis | C1 | C2 | C3 | C4 | verdict by the declared rule |
|---|---|---|---|---|---|
| h1a w_ft 0.7 on CS >= 0.45 (else 0.5), vs PLAIN | +0.00002 +- 0.00007 | +0.00014 +- 0.00009 | +0.00002 +- 0.00005 | +0.00000 +- 0.00008 | NOT SHOWN (below +1 SE on C1) |
| h1a on top of the M-rule, vs M-RULE | +0.00000 +- 0.00006 | +0.00010 +- 0.00009 | -0.00001 +- 0.00007 | +0.00001 +- 0.00007 | NOT SHOWN |
| h1b w_ft 0.7 on CL >= 0.45, vs PLAIN | +0.00008 +- 0.00013 | +0.00018 +- 0.00014 | -0.00010 +- 0.00012 | +0.00014 +- 0.00013 | NOT SHOWN |
| h1b on top of the M-rule, vs M-RULE | +0.00010 +- 0.00013 | +0.00014 +- 0.00014 | -0.00023 +- 0.00012 | -0.00008 +- 0.00012 | NOT SHOWN |
| h2 M-rule 0.2 / 0.5 minus 0.2 / 0.4, 8-member pools (4 wide + 4 d128) | FD16 + 8: +0.00013 +- 0.00014 | FD8 + 8: -0.00019 +- 0.00014 | W16 + 8: -0.00020 +- 0.00014 | W8 + 8: -0.00017 +- 0.00014 | REFUTED by the declared rule (W16 + 8 at -1.41 SE, pmix-private -0.00042 +- 0.00019); the FD16 + 8 leg is +0.9 SE |
| (same pools) M-rule 0.2 / 0.4 minus PLAIN | -0.00010 +- 0.00020 | +0.00047 +- 0.00020 | +0.00064 +- 0.00021 | +0.00037 +- 0.00021 | fixed rule, new pool members: 3 of 4 positive |
| (ref) the same contrast, 4-member pools | +0.00006 +- 0.00013 | -0.00011 +- 0.00013 | +0.00009 +- 0.00014 | +0.00010 +- 0.00015 | flat |
| h3-TO M-rule in the train-only branch only, vs M-RULE | -0.00009 +- 0.00015 | -0.00017 +- 0.00014 | -0.00002 +- 0.00014 | -0.00024 +- 0.00013 | does not replace; share of the gain 0.67 / 0.67 / 0.97 / 0.45 |
| h3-FT M-rule in the fine-tuned branch only, vs M-RULE | -0.00017 +- 0.00017 | -0.00037 +- 0.00016 | -0.00032 +- 0.00016 | -0.00032 +- 0.00013 | does not replace; share 0.36 / 0.28 / 0.55 / 0.26 |
| h4a three-step (0.2, 0.3, 0.4) on M < 30 / 30-99 / >= 100, vs M-RULE | +0.00016 +- 0.00014 | -0.00004 +- 0.00013 | -0.00016 +- 0.00013 | -0.00019 +- 0.00012 | NOT SHOWN (0 of 3 replications) |
| h4b three-step (0.1, 0.3, 0.5), vs M-RULE | +0.00036 +- 0.00022 | -0.00011 +- 0.00021 | -0.00027 +- 0.00022 | -0.00005 +- 0.00021 | NOT SHOWN (0 of 3) |

h1: the cold cell alone reads +0.00023 +- 0.00088 (C1), +0.00166 +- 0.00110 (C2), +0.00028 +- 0.00065 (C3), +0.00005 +- 0.00098 (C4) per cold feed with the segment definition: val has 1,213 such feeds and cannot resolve an effect of the size that would matter at the private mix (a third of the feeds). Not refuted, not shown. h3: both branches contribute, the train-only branch more; the sum of the two single-branch gains is about the gain of the full rule. Full log:

```
cells: CS>=.45 0.0841 CL>=.45 0.2592 | M<30 0.2692 30-99 0.36 >=100 0.3708
=== C1 ('FD16', 'FDP'): plain 0.40271 | MRULE - plain +0.00026+-0.00021
 h1a CS>=.45 c .7/.5 on a=.2  minus PLAIN: +0.00002+-0.00007 (z +0.27) | pmix-public +0.00001+-0.00014 private +0.00002+-0.00021 | M<50 +0.00001+-0.00015 M>=50 +0.00003+-0.00007 | cold cell only +0.00023+-0.00088
 h1a CS>=.45 c .7/.5 on MRULE minus MRULE: +0.00000+-0.00006 (z +0.04) | pmix-public -0.00005+-0.00011 private -0.00007+-0.00016 | M<50 +0.00001+-0.00015 M>=50 -0.00000+-0.00003 | cold cell only +0.00003+-0.00077
 h1b CL>=.45 c .7/.5 on a=.2  minus PLAIN: +0.00008+-0.00013 (z +0.60) | pmix-public +0.00010+-0.00020 private +0.00008+-0.00027 | M<50 +0.00022+-0.00025 M>=50 -0.00003+-0.00013 | cold cell only +0.00029+-0.00049
 h1b CL>=.45 c .7/.5 on MRULE minus MRULE: +0.00010+-0.00013 (z +0.77) | pmix-public +0.00010+-0.00020 private +0.00003+-0.00025 | M<50 +0.00022+-0.00025 M>=50 +0.00002+-0.00015 | cold cell only +0.00040+-0.00052
 (ref, not a hypothesis) c = .7 everywhere on a=.2 minus PLAIN: -0.00038+-0.00024 (z -1.56) | pmix-public -0.00010+-0.00029 private -0.00002+-0.00036 | M<50 -0.00014+-0.00046 M>=50 -0.00054+-0.00026
 h3-TO (M-rule in the train-only branch only) minus PLAIN: +0.00018+-0.00016 (z +1.12) | pmix-public +0.00037+-0.00021 private +0.00053+-0.00027 | M<50 +0.00000+-0.00000 M>=50 +0.00030+-0.00027
 h3-TO minus MRULE: -0.00009+-0.00015 (z -0.56) | pmix-public +0.00010+-0.00021 private +0.00016+-0.00026 | M<50 +0.00000+-0.00000 M>=50 -0.00014+-0.00026
 h3-FT (M-rule in the fine-tuned branch only) minus PLAIN: +0.00009+-0.00014 (z +0.70) | pmix-public +0.00013+-0.00019 private +0.00018+-0.00024 | M<50 +0.00000+-0.00000 M>=50 +0.00016+-0.00023
 h3-FT minus MRULE: -0.00017+-0.00017 (z -0.98) | pmix-public -0.00014+-0.00022 private -0.00020+-0.00028 | M<50 +0.00000+-0.00000 M>=50 -0.00028+-0.00029
   share of the MRULE gain carried: TO-only 0.67  FT-only 0.36
 h4a (.2,.3,.4) minus PLAIN: +0.00043+-0.00019 (z +2.24) | pmix-public +0.00047+-0.00025 private +0.00062+-0.00030 | M<50 +0.00024+-0.00019 M>=50 +0.00056+-0.00029
 h4a (.2,.3,.4) minus MRULE: +0.00016+-0.00014 (z +1.17) | pmix-public +0.00021+-0.00019 private +0.00025+-0.00024 | M<50 +0.00024+-0.00019 M>=50 +0.00011+-0.00020 | by cell M<30 / 30-99 / >=100: +0.00000+-0.00000 / +0.00046+-0.00039 / +0.00000+-0.00000
 h4b (.1,.3,.5) minus PLAIN: +0.00062+-0.00026 (z +2.38) | pmix-public +0.00066+-0.00034 private +0.00081+-0.00041 | M<50 +0.00071+-0.00041 M>=50 +0.00056+-0.00034
 h4b (.1,.3,.5) minus MRULE: +0.00036+-0.00022 (z +1.60) | pmix-public +0.00039+-0.00029 private +0.00043+-0.00036 | M<50 +0.00071+-0.00041 M>=50 +0.00012+-0.00025 | by cell M<30 / 30-99 / >=100: +0.00071+-0.00055 / +0.00046+-0.00039 / +0.00000+-0.00024
=== C2 ('FD8', 'FDP'): plain 0.40240 | MRULE - plain +0.00051+-0.00021
 h1a CS>=.45 c .7/.5 on a=.2  minus PLAIN: +0.00014+-0.00009 (z +1.50) | pmix-public +0.00023+-0.00017 private +0.00033+-0.00025 | M<50 +0.00034+-0.00021 M>=50 -0.00000+-0.00006 | cold cell only +0.00166+-0.00110
 h1a CS>=.45 c .7/.5 on MRULE minus MRULE: +0.00010+-0.00009 (z +1.09) | pmix-public +0.00009+-0.00017 private +0.00010+-0.00024 | M<50 +0.00034+-0.00021 M>=50 -0.00007+-0.00006 | cold cell only +0.00121+-0.00111
 h1b CL>=.45 c .7/.5 on a=.2  minus PLAIN: +0.00018+-0.00014 (z +1.33) | pmix-public +0.00035+-0.00022 private +0.00045+-0.00030 | M<50 +0.00040+-0.00029 M>=50 +0.00003+-0.00012 | cold cell only +0.00071+-0.00053
 h1b CL>=.45 c .7/.5 on MRULE minus MRULE: +0.00014+-0.00014 (z +1.05) | pmix-public +0.00016+-0.00022 private +0.00014+-0.00030 | M<50 +0.00040+-0.00029 M>=50 -0.00004+-0.00012 | cold cell only +0.00055+-0.00053
 (ref, not a hypothesis) c = .7 everywhere on a=.2 minus PLAIN: +0.00004+-0.00025 (z +0.17) | pmix-public +0.00029+-0.00032 private +0.00043+-0.00039 | M<50 +0.00018+-0.00046 M>=50 -0.00005+-0.00027
 h3-TO (M-rule in the train-only branch only) minus PLAIN: +0.00034+-0.00016 (z +2.15) | pmix-public +0.00050+-0.00023 private +0.00059+-0.00029 | M<50 +0.00000+-0.00000 M>=50 +0.00058+-0.00027
 h3-TO minus MRULE: -0.00017+-0.00014 (z -1.19) | pmix-public -0.00009+-0.00018 private -0.00013+-0.00022 | M<50 +0.00000+-0.00000 M>=50 -0.00028+-0.00024
 h3-FT (M-rule in the fine-tuned branch only) minus PLAIN: +0.00014+-0.00014 (z +1.03) | pmix-public +0.00005+-0.00020 private +0.00003+-0.00026 | M<50 +0.00000+-0.00000 M>=50 +0.00024+-0.00023
 h3-FT minus MRULE: -0.00037+-0.00016 (z -2.30) | pmix-public -0.00054+-0.00020 private -0.00069+-0.00024 | M<50 +0.00000+-0.00000 M>=50 -0.00062+-0.00027
   share of the MRULE gain carried: TO-only 0.67  FT-only 0.28
 h4a (.2,.3,.4) minus PLAIN: +0.00047+-0.00019 (z +2.44) | pmix-public +0.00053+-0.00026 private +0.00067+-0.00032 | M<50 +0.00010+-0.00018 M>=50 +0.00072+-0.00030
 h4a (.2,.3,.4) minus MRULE: -0.00004+-0.00013 (z -0.31) | pmix-public -0.00006+-0.00015 private -0.00005+-0.00018 | M<50 +0.00010+-0.00018 M>=50 -0.00014+-0.00018 | by cell M<30 / 30-99 / >=100: +0.00000+-0.00000 / -0.00011+-0.00036 / +0.00000+-0.00000
 h4b (.1,.3,.5) minus PLAIN: +0.00040+-0.00026 (z +1.56) | pmix-public +0.00052+-0.00033 private +0.00065+-0.00041 | M<50 -0.00007+-0.00037 M>=50 +0.00072+-0.00035
 h4b (.1,.3,.5) minus MRULE: -0.00011+-0.00021 (z -0.53) | pmix-public -0.00007+-0.00025 private -0.00007+-0.00030 | M<50 -0.00007+-0.00037 M>=50 -0.00014+-0.00025 | by cell M<30 / 30-99 / >=100: -0.00026+-0.00048 / -0.00011+-0.00036 / -0.00000+-0.00027
=== C3 ('W16', 'P1w'): plain 0.40214 | MRULE - plain +0.00071+-0.00021
 h1a CS>=.45 c .7/.5 on a=.2  minus PLAIN: +0.00002+-0.00005 (z +0.42) | pmix-public +0.00003+-0.00011 private +0.00004+-0.00016 | M<50 +0.00005+-0.00010 M>=50 +0.00000+-0.00006 | cold cell only +0.00028+-0.00065
 h1a CS>=.45 c .7/.5 on MRULE minus MRULE: -0.00001+-0.00007 (z -0.11) | pmix-public -0.00008+-0.00015 private -0.00012+-0.00023 | M<50 +0.00005+-0.00010 M>=50 -0.00005+-0.00010 | cold cell only -0.00010+-0.00084
 h1b CL>=.45 c .7/.5 on a=.2  minus PLAIN: -0.00010+-0.00012 (z -0.78) | pmix-public -0.00005+-0.00019 private -0.00003+-0.00025 | M<50 -0.00035+-0.00022 M>=50 +0.00008+-0.00014 | cold cell only -0.00037+-0.00048
 h1b CL>=.45 c .7/.5 on MRULE minus MRULE: -0.00023+-0.00012 (z -1.94) | pmix-public -0.00027+-0.00020 private -0.00029+-0.00027 | M<50 -0.00035+-0.00022 M>=50 -0.00015+-0.00013 | cold cell only -0.00089+-0.00046
 (ref, not a hypothesis) c = .7 everywhere on a=.2 minus PLAIN: -0.00015+-0.00024 (z -0.64) | pmix-public -0.00018+-0.00030 private -0.00018+-0.00036 | M<50 -0.00058+-0.00040 M>=50 +0.00014+-0.00030
 h3-TO (M-rule in the train-only branch only) minus PLAIN: +0.00069+-0.00016 (z +4.34) | pmix-public +0.00067+-0.00019 private +0.00064+-0.00023 | M<50 +0.00000+-0.00000 M>=50 +0.00116+-0.00027
 h3-TO minus MRULE: -0.00002+-0.00014 (z -0.12) | pmix-public -0.00014+-0.00021 private -0.00029+-0.00027 | M<50 +0.00000+-0.00000 M>=50 -0.00003+-0.00025
 h3-FT (M-rule in the fine-tuned branch only) minus PLAIN: +0.00038+-0.00013 (z +2.86) | pmix-public +0.00046+-0.00017 private +0.00052+-0.00022 | M<50 +0.00000+-0.00000 M>=50 +0.00065+-0.00023
 h3-FT minus MRULE: -0.00032+-0.00016 (z -2.04) | pmix-public -0.00035+-0.00021 private -0.00041+-0.00026 | M<50 +0.00000+-0.00000 M>=50 -0.00054+-0.00027
   share of the MRULE gain carried: TO-only 0.97  FT-only 0.55
 h4a (.2,.3,.4) minus PLAIN: +0.00055+-0.00020 (z +2.75) | pmix-public +0.00060+-0.00027 private +0.00068+-0.00034 | M<50 -0.00012+-0.00021 M>=50 +0.00101+-0.00030
 h4a (.2,.3,.4) minus MRULE: -0.00016+-0.00013 (z -1.17) | pmix-public -0.00021+-0.00017 private -0.00025+-0.00021 | M<50 -0.00012+-0.00021 M>=50 -0.00018+-0.00017 | by cell M<30 / 30-99 / >=100: +0.00000+-0.00000 / -0.00043+-0.00037 / +0.00000+-0.00000
 h4b (.1,.3,.5) minus PLAIN: +0.00044+-0.00027 (z +1.65) | pmix-public +0.00054+-0.00035 private +0.00066+-0.00044 | M<50 -0.00044+-0.00040 M>=50 +0.00104+-0.00036
 h4b (.1,.3,.5) minus MRULE: -0.00027+-0.00022 (z -1.23) | pmix-public -0.00027+-0.00027 private -0.00028+-0.00034 | M<50 -0.00044+-0.00040 M>=50 -0.00015+-0.00024 | by cell M<30 / 30-99 / >=100: -0.00048+-0.00051 / -0.00043+-0.00037 / +0.00005+-0.00027
=== C4 ('W8', 'P1w'): plain 0.40218 | MRULE - plain +0.00043+-0.00019
 h1a CS>=.45 c .7/.5 on a=.2  minus PLAIN: +0.00000+-0.00008 (z +0.05) | pmix-public -0.00001+-0.00017 private +0.00000+-0.00025 | M<50 +0.00007+-0.00014 M>=50 -0.00004+-0.00010 | cold cell only +0.00005+-0.00098
 h1a CS>=.45 c .7/.5 on MRULE minus MRULE: +0.00001+-0.00007 (z +0.16) | pmix-public +0.00000+-0.00013 private -0.00001+-0.00019 | M<50 +0.00007+-0.00014 M>=50 -0.00003+-0.00006 | cold cell only +0.00013+-0.00079
 h1b CL>=.45 c .7/.5 on a=.2  minus PLAIN: +0.00014+-0.00013 (z +1.08) | pmix-public +0.00026+-0.00022 private +0.00027+-0.00030 | M<50 +0.00006+-0.00025 M>=50 +0.00020+-0.00014 | cold cell only +0.00055+-0.00051
 h1b CL>=.45 c .7/.5 on MRULE minus MRULE: -0.00008+-0.00012 (z -0.62) | pmix-public -0.00014+-0.00019 private -0.00016+-0.00025 | M<50 +0.00006+-0.00025 M>=50 -0.00017+-0.00011 | cold cell only -0.00029+-0.00046
 (ref, not a hypothesis) c = .7 everywhere on a=.2 minus PLAIN: -0.00010+-0.00024 (z -0.40) | pmix-public +0.00011+-0.00032 private +0.00012+-0.00040 | M<50 -0.00040+-0.00045 M>=50 +0.00011+-0.00027
 h3-TO (M-rule in the train-only branch only) minus PLAIN: +0.00019+-0.00015 (z +1.28) | pmix-public +0.00034+-0.00021 private +0.00040+-0.00027 | M<50 +0.00000+-0.00000 M>=50 +0.00032+-0.00025
 h3-TO minus MRULE: -0.00024+-0.00013 (z -1.82) | pmix-public -0.00042+-0.00017 private -0.00052+-0.00021 | M<50 +0.00000+-0.00000 M>=50 -0.00040+-0.00022
 h3-FT (M-rule in the fine-tuned branch only) minus PLAIN: +0.00011+-0.00014 (z +0.78) | pmix-public +0.00025+-0.00019 private +0.00028+-0.00024 | M<50 +0.00000+-0.00000 M>=50 +0.00019+-0.00024
 h3-FT minus MRULE: -0.00032+-0.00013 (z -2.41) | pmix-public -0.00052+-0.00018 private -0.00064+-0.00023 | M<50 +0.00000+-0.00000 M>=50 -0.00053+-0.00022
   share of the MRULE gain carried: TO-only 0.45  FT-only 0.26
 h4a (.2,.3,.4) minus PLAIN: +0.00023+-0.00019 (z +1.26) | pmix-public +0.00052+-0.00025 private +0.00064+-0.00032 | M<50 -0.00023+-0.00019 M>=50 +0.00056+-0.00029
 h4a (.2,.3,.4) minus MRULE: -0.00019+-0.00012 (z -1.67) | pmix-public -0.00025+-0.00015 private -0.00028+-0.00019 | M<50 -0.00023+-0.00019 M>=50 -0.00016+-0.00014 | by cell M<30 / 30-99 / >=100: +0.00000+-0.00000 / -0.00053+-0.00032 / +0.00000+-0.00000
 h4b (.1,.3,.5) minus PLAIN: +0.00038+-0.00026 (z +1.46) | pmix-public +0.00055+-0.00034 private +0.00062+-0.00042 | M<50 +0.00015+-0.00039 M>=50 +0.00054+-0.00035
 h4b (.1,.3,.5) minus MRULE: -0.00005+-0.00021 (z -0.22) | pmix-public -0.00022+-0.00026 private -0.00029+-0.00032 | M<50 +0.00015+-0.00039 M>=50 -0.00018+-0.00023 | by cell M<30 / 30-99 / >=100: +0.00058+-0.00052 / -0.00053+-0.00032 / -0.00003+-0.00028
=== h2: M-rule 0.2 / 0.5 vs 0.2 / 0.4 with 8 popularity-light members
 W16 + 4 wide + 4 d128 (W16P20x8): plain 0.40189
    rule .2/.4 minus PLAIN: +0.00064+-0.00021 (z +3.06) | pmix-public +0.00091+-0.00029 private +0.00120+-0.00037 | M<50 +0.00000+-0.00000 M>=50 +0.00107+-0.00035
    rule .2/.5 minus PLAIN: +0.00044+-0.00025 (z +1.73) | pmix-public +0.00054+-0.00033 private +0.00079+-0.00041 | M<50 +0.00000+-0.00000 M>=50 +0.00074+-0.00043
    rule .2/.5 minus rule .2/.4: -0.00020+-0.00014 (z -1.41) | pmix-public -0.00037+-0.00016 private -0.00042+-0.00019 | M<50 +0.00000+-0.00000 M>=50 -0.00034+-0.00024
 W8 + 4 wide + 4 d128 (W8P20x8): plain 0.40224
    rule .2/.4 minus PLAIN: +0.00037+-0.00021 (z +1.81) | pmix-public +0.00058+-0.00027 private +0.00070+-0.00034 | M<50 +0.00000+-0.00000 M>=50 +0.00063+-0.00035
    rule .2/.5 minus PLAIN: +0.00021+-0.00026 (z +0.81) | pmix-public +0.00034+-0.00033 private +0.00043+-0.00041 | M<50 +0.00000+-0.00000 M>=50 +0.00035+-0.00043
    rule .2/.5 minus rule .2/.4: -0.00017+-0.00014 (z -1.15) | pmix-public -0.00024+-0.00017 private -0.00027+-0.00020 | M<50 +0.00000+-0.00000 M>=50 -0.00028+-0.00024
 FD16 + 8 wide full-data (FD16P20w8): NOT READ, files missing for ['abl_fd10_pop1kd05w192_s955', 'abl_fd10_pop1kd05w192_s956', 'abl_fd10_pop1kd05w192_s957', 'abl_fd10_pop1kd05w192_s958']
 FD16 + 4 wide + 4 d128 full-data (FD16P20x8): plain 0.40272
    rule .2/.4 minus PLAIN: -0.00010+-0.00020 (z -0.51) | pmix-public -0.00033+-0.00027 private -0.00035+-0.00034 | M<50 +0.00000+-0.00000 M>=50 -0.00018+-0.00034
    rule .2/.5 minus PLAIN: +0.00003+-0.00025 (z +0.11) | pmix-public -0.00009+-0.00034 private -0.00007+-0.00043 | M<50 +0.00000+-0.00000 M>=50 +0.00005+-0.00043
    rule .2/.5 minus rule .2/.4: +0.00013+-0.00014 (z +0.90) | pmix-public +0.00024+-0.00019 private +0.00028+-0.00023 | M<50 +0.00000+-0.00000 M>=50 +0.00022+-0.00025
 FD8 + 8 wide full-data: NOT READ, files missing for ['abl_fd10_pop1kd05w192_s955', 'abl_fd10_pop1kd05w192_s956', 'abl_fd10_pop1kd05w192_s957', 'abl_fd10_pop1kd05w192_s958']
 FD8 + 4 wide + 4 d128 full-data: plain 0.40238
    rule .2/.4 minus PLAIN: +0.00047+-0.00020 (z +2.34) | pmix-public +0.00041+-0.00027 private +0.00046+-0.00033 | M<50 +0.00000+-0.00000 M>=50 +0.00080+-0.00034
    rule .2/.5 minus PLAIN: +0.00028+-0.00026 (z +1.09) | pmix-public +0.00009+-0.00034 private +0.00007+-0.00042 | M<50 +0.00000+-0.00000 M>=50 +0.00047+-0.00043
    rule .2/.5 minus rule .2/.4: -0.00019+-0.00014 (z -1.32) | pmix-public -0.00032+-0.00019 private -0.00039+-0.00023 | M<50 +0.00000+-0.00000 M>=50 -0.00032+-0.00025
 (ref, 4-member pool) C1 rule .2/.5 minus rule .2/.4: +0.00006+-0.00013 (z +0.43) | pmix-public +0.00009+-0.00018 private +0.00008+-0.00024 | M<50 +0.00000+-0.00000 M>=50 +0.00010+-0.00023
 (ref, 4-member pool) C2 rule .2/.5 minus rule .2/.4: -0.00011+-0.00013 (z -0.85) | pmix-public -0.00016+-0.00018 private -0.00026+-0.00023 | M<50 +0.00000+-0.00000 M>=50 -0.00019+-0.00022
 (ref, 4-member pool) C3 rule .2/.5 minus rule .2/.4: +0.00009+-0.00014 (z +0.68) | pmix-public +0.00010+-0.00018 private +0.00015+-0.00023 | M<50 +0.00000+-0.00000 M>=50 +0.00016+-0.00023
 (ref, 4-member pool) C4 rule .2/.5 minus rule .2/.4: +0.00010+-0.00015 (z +0.64) | pmix-public +0.00005+-0.00019 private +0.00001+-0.00022 | M<50 +0.00000+-0.00000 M>=50 +0.00016+-0.00025
```

## 6. Robustness (step 5) [M]

Nested gains over PLAIN by day, by M bucket, under pmix weights (fixed M-rule, the M-rule family, and three other families for contrast):

```
=== C1 ('FD16', 'FDP')  (nested per-feed vectors, mean over 5 fold seeds; minus PLAIN unless stated)
rule         | val                 | pmix-public         | pmix-private        | day 1 (n 7543)      | day 2 (n 6880)      | M5-10 (n 1214) | M11-25 (n 2155) | M26-50 (n 2610) | M51-100 (n 3138) | M101-200 (n 2985) | M201-300 (n 2321)
fixed MRULE  | +0.00026+-0.00021 | +0.00027+-0.00029 | +0.00038+-0.00036 | +0.00052+-0.00030 | -0.00002+-0.00030 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00009+-0.00007 | +0.00039+-0.00071 | +0.00061+-0.00060 | +0.00021+-0.00046
C(M50)       | +0.00018+-0.00030 | +0.00027+-0.00039 | +0.00040+-0.00049 | +0.00045+-0.00042 | -0.00011+-0.00041 | +0.00025+-0.00092 | -0.00055+-0.00069 | +0.00054+-0.00056 | +0.00022+-0.00084 | -0.00011+-0.00070 | +0.00075+-0.00052
G(M3q)       | +0.00031+-0.00029 | +0.00037+-0.00038 | +0.00047+-0.00048 | +0.00072+-0.00042 | -0.00014+-0.00040 | -0.00009+-0.00070 | -0.00039+-0.00064 | +0.00118+-0.00072 | +0.00048+-0.00086 | +0.00013+-0.00061 | +0.00018+-0.00042
K(Mq,Hq)     | +0.00047+-0.00026 | +0.00050+-0.00034 | +0.00051+-0.00042 | +0.00072+-0.00037 | +0.00020+-0.00037 | -0.00068+-0.00089 | -0.00088+-0.00072 | +0.00074+-0.00059 | +0.00126+-0.00060 | +0.00073+-0.00063 | +0.00064+-0.00041
F(Hq)        | +0.00076+-0.00032 | +0.00108+-0.00042 | +0.00129+-0.00051 | +0.00091+-0.00044 | +0.00060+-0.00046 | +0.00193+-0.00129 | +0.00018+-0.00098 | +0.00091+-0.00083 | +0.00085+-0.00071 | +0.00067+-0.00062 | +0.00053+-0.00050
  minus fixed MRULE at the private mix: C(M50) +0.00002+-0.00033 | C(Mq) +0.00005+-0.00036 | G(M3q) +0.00009+-0.00033 | G(M3r) -0.00004+-0.00032 | K(Mq,Hq) +0.00013+-0.00038 | F(Hq) +0.00091+-0.00040 | A +0.00039+-0.00038 | VAR B-3 +0.00004+-0.00031 | VAR B-all -0.00015+-0.00030
=== C2 ('FD8', 'FDP')  (nested per-feed vectors, mean over 5 fold seeds; minus PLAIN unless stated)
rule         | val                 | pmix-public         | pmix-private        | day 1 (n 7543)      | day 2 (n 6880)      | M5-10 (n 1214) | M11-25 (n 2155) | M26-50 (n 2610) | M51-100 (n 3138) | M101-200 (n 2985) | M201-300 (n 2321)
fixed MRULE  | +0.00051+-0.00021 | +0.00059+-0.00028 | +0.00072+-0.00034 | +0.00041+-0.00029 | +0.00061+-0.00031 | +0.00000+-0.00000 | +0.00000+-0.00000 | -0.00001+-0.00007 | +0.00155+-0.00068 | +0.00045+-0.00058 | +0.00049+-0.00052
C(M50)       | +0.00021+-0.00025 | +0.00043+-0.00032 | +0.00062+-0.00039 | +0.00022+-0.00034 | +0.00021+-0.00038 | -0.00103+-0.00081 | -0.00081+-0.00062 | -0.00049+-0.00048 | +0.00155+-0.00068 | +0.00045+-0.00058 | +0.00049+-0.00052
G(M3q)       | +0.00010+-0.00025 | +0.00019+-0.00032 | +0.00026+-0.00040 | +0.00002+-0.00034 | +0.00020+-0.00036 | -0.00083+-0.00054 | -0.00059+-0.00036 | +0.00007+-0.00052 | +0.00105+-0.00072 | -0.00011+-0.00063 | +0.00026+-0.00051
K(Mq,Hq)     | +0.00028+-0.00028 | +0.00026+-0.00036 | +0.00031+-0.00046 | +0.00025+-0.00038 | +0.00032+-0.00041 | -0.00024+-0.00113 | +0.00083+-0.00074 | -0.00087+-0.00049 | +0.00079+-0.00061 | +0.00038+-0.00072 | +0.00053+-0.00058
F(Hq)        | +0.00041+-0.00032 | +0.00075+-0.00041 | +0.00094+-0.00051 | +0.00059+-0.00045 | +0.00022+-0.00046 | +0.00001+-0.00140 | -0.00008+-0.00102 | +0.00026+-0.00081 | +0.00127+-0.00064 | -0.00014+-0.00064 | +0.00080+-0.00054
  minus fixed MRULE at the private mix: C(M50) -0.00010+-0.00020 | C(Mq) -0.00042+-0.00028 | G(M3q) -0.00046+-0.00022 | G(M3r) -0.00016+-0.00025 | K(Mq,Hq) -0.00041+-0.00034 | F(Hq) +0.00022+-0.00043 | A +0.00008+-0.00042 | VAR B-3 -0.00016+-0.00032 | VAR B-all +0.00026+-0.00036
=== C3 ('W16', 'P1w')  (nested per-feed vectors, mean over 5 fold seeds; minus PLAIN unless stated)
rule         | val                 | pmix-public         | pmix-private        | day 1 (n 7543)      | day 2 (n 6880)      | M5-10 (n 1214) | M11-25 (n 2155) | M26-50 (n 2610) | M51-100 (n 3138) | M101-200 (n 2985) | M201-300 (n 2321)
fixed MRULE  | +0.00071+-0.00021 | +0.00081+-0.00027 | +0.00093+-0.00034 | +0.00079+-0.00030 | +0.00062+-0.00028 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00010+-0.00009 | +0.00164+-0.00064 | +0.00051+-0.00065 | +0.00140+-0.00044
C(M50)       | +0.00055+-0.00026 | +0.00070+-0.00034 | +0.00090+-0.00043 | +0.00073+-0.00039 | +0.00034+-0.00034 | -0.00083+-0.00046 | -0.00020+-0.00033 | -0.00034+-0.00024 | +0.00180+-0.00077 | +0.00024+-0.00076 | +0.00166+-0.00057
G(M3q)       | +0.00044+-0.00027 | +0.00050+-0.00035 | +0.00060+-0.00043 | +0.00096+-0.00039 | -0.00013+-0.00037 | -0.00012+-0.00013 | +0.00003+-0.00009 | -0.00089+-0.00063 | +0.00182+-0.00085 | -0.00007+-0.00067 | +0.00138+-0.00048
K(Mq,Hq)     | +0.00018+-0.00024 | +0.00035+-0.00032 | +0.00053+-0.00041 | +0.00031+-0.00035 | +0.00003+-0.00032 | -0.00165+-0.00070 | -0.00052+-0.00042 | -0.00082+-0.00036 | +0.00147+-0.00057 | -0.00003+-0.00075 | +0.00143+-0.00049
F(Hq)        | -0.00019+-0.00028 | -0.00032+-0.00035 | -0.00034+-0.00043 | +0.00002+-0.00039 | -0.00042+-0.00041 | -0.00133+-0.00111 | -0.00145+-0.00085 | -0.00201+-0.00074 | +0.00144+-0.00057 | -0.00019+-0.00061 | +0.00140+-0.00044
  minus fixed MRULE at the private mix: C(M50) -0.00003+-0.00023 | C(Mq) -0.00051+-0.00025 | G(M3q) -0.00033+-0.00028 | G(M3r) -0.00073+-0.00024 | K(Mq,Hq) -0.00040+-0.00032 | F(Hq) -0.00127+-0.00036 | A -0.00097+-0.00036 | VAR B-3 -0.00053+-0.00027 | VAR B-all -0.00070+-0.00026
=== C4 ('W8', 'P1w')  (nested per-feed vectors, mean over 5 fold seeds; minus PLAIN unless stated)
rule         | val                 | pmix-public         | pmix-private        | day 1 (n 7543)      | day 2 (n 6880)      | M5-10 (n 1214) | M11-25 (n 2155) | M26-50 (n 2610) | M51-100 (n 3138) | M101-200 (n 2985) | M201-300 (n 2321)
fixed MRULE  | +0.00043+-0.00019 | +0.00076+-0.00026 | +0.00092+-0.00033 | +0.00041+-0.00029 | +0.00044+-0.00024 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00005+-0.00008 | +0.00040+-0.00060 | +0.00053+-0.00056 | +0.00136+-0.00043
C(M50)       | +0.00028+-0.00029 | +0.00057+-0.00037 | +0.00070+-0.00046 | +0.00018+-0.00042 | +0.00038+-0.00038 | -0.00138+-0.00093 | +0.00057+-0.00060 | -0.00087+-0.00048 | +0.00085+-0.00079 | +0.00011+-0.00074 | +0.00161+-0.00053
G(M3q)       | +0.00028+-0.00028 | +0.00070+-0.00036 | +0.00091+-0.00044 | +0.00023+-0.00040 | +0.00034+-0.00038 | -0.00157+-0.00093 | +0.00137+-0.00063 | -0.00116+-0.00063 | +0.00058+-0.00079 | +0.00039+-0.00059 | +0.00133+-0.00044
K(Mq,Hq)     | +0.00038+-0.00023 | +0.00052+-0.00031 | +0.00060+-0.00039 | +0.00041+-0.00035 | +0.00034+-0.00030 | -0.00036+-0.00058 | +0.00074+-0.00057 | -0.00055+-0.00046 | +0.00092+-0.00057 | +0.00021+-0.00065 | +0.00095+-0.00045
F(Hq)        | -0.00034+-0.00022 | -0.00025+-0.00028 | -0.00022+-0.00036 | -0.00025+-0.00032 | -0.00045+-0.00030 | +0.00017+-0.00088 | -0.00059+-0.00068 | -0.00161+-0.00057 | -0.00029+-0.00046 | +0.00014+-0.00046 | +0.00034+-0.00034
  minus fixed MRULE at the private mix: C(M50) -0.00022+-0.00030 | C(Mq) -0.00025+-0.00023 | G(M3q) -0.00000+-0.00031 | G(M3r) -0.00050+-0.00028 | K(Mq,Hq) -0.00031+-0.00029 | F(Hq) -0.00113+-0.00033 | A -0.00120+-0.00033 | VAR B-3 -0.00077+-0.00026 | VAR B-all -0.00038+-0.00029
```

- The fixed M-rule is positive in every long-feed bucket of every composition (12 cells of 12). By day it is +0.00052 / -0.00002 on C1 and balanced on the other three (+0.00041 / +0.00061, +0.00079 / +0.00062, +0.00041 / +0.00044): no day pattern.
- Under the public / private mix the M-rule family reads higher than on plain val in all four compositions (long and cold feeds weigh more on test).

Label-free on test (gate cell shares; ranking change against the PLAIN file of the same members):

```
=== LABEL-FREE gate coverage: share of feeds per cell, val | public-like test | private-like test   (cell 0 / 1 / 2)
M50   0.409/0.591 | 0.387/0.613 | 0.392/0.608
Mq    0.511/0.489 | 0.487/0.513 | 0.492/0.508
CS45  0.916/0.084 | 0.785/0.215 | 0.673/0.327
CLq   0.665/0.335 | 0.493/0.507 | 0.384/0.616
Hq    0.496/0.504 | 0.498/0.502 | 0.501/0.499
U0    0.954/0.046 | 0.959/0.041 | 0.959/0.041
DAY   0.523/0.477 | 0.000/1.000 | 0.000/1.000
M3q   0.344/0.337/0.319 | 0.315/0.332/0.353 | 0.325/0.337/0.338
M3r   0.269/0.360/0.371 | 0.242/0.356/0.403 | 0.250/0.361/0.389
CL3q  0.493/0.306/0.201 | 0.318/0.351/0.331 | 0.225/0.346/0.429
H3q   0.336/0.327/0.337 | 0.326/0.337/0.337 | 0.333/0.336/0.330
ranges: M val [5, 299] test [5, 300] ; H val [0, 256] test [0, 256] ; CL val [0, 1] test [0, 1] ; CS val [0, 1] test [0, 1]
joint M50 x Hq (cells 00/01/10/11): 0.200/0.208/0.295/0.296 | 0.190/0.197/0.308/0.305 | 0.194/0.198/0.308/0.301
joint Mq x Hq (cells 00/01/10/11): 0.252/0.260/0.244/0.245 | 0.240/0.246/0.258/0.256 | 0.244/0.248/0.257/0.251
=== LABEL-FREE test ranking change vs the PLAIN file of the same members, C1 ('FD16', 'FDP'): share of feeds with the same top-1 | mean per-feed Spearman   (all / public-like / private-like)
 MRULE (a .2 | .4 by M>=50)                      : same top-1 0.9564 / 0.9542 / 0.9574 | Spearman 0.9993 / 0.9993 / 0.9993 | feeds with any order change 0.610
 M-rule .2 | .5                                  : same top-1 0.9363 / 0.9324 / 0.9380 | Spearman 0.9987 / 0.9987 / 0.9987 | feeds with any order change 0.610
 h3-TO (M-rule in the train-only branch only)    : same top-1 0.9734 / 0.9720 / 0.9739 | Spearman 0.9997 / 0.9997 / 0.9997 | feeds with any order change 0.610
 h4a (.2,.3,.4 by M 30/100)                      : same top-1 0.9596 / 0.9577 / 0.9604 | Spearman 0.9994 / 0.9993 / 0.9994 | feeds with any order change 0.744
 F(Hq) best on C1 (a .4; c .7 if H<190 else .5)  : same top-1 0.9333 / 0.9314 / 0.9342 | Spearman 0.9974 / 0.9975 / 0.9973 | feeds with any order change 0.908
 CONST a .4 c .5                                 : same top-1 0.9380 / 0.9374 / 0.9383 | Spearman 0.9977 / 0.9979 / 0.9976 | feeds with any order change 0.902
 CONST a .4 c .7 (A best on C1)                  : same top-1 0.9287 / 0.9247 / 0.9304 | Spearman 0.9970 / 0.9971 / 0.9969 | feeds with any order change 0.914
=== LABEL-FREE test ranking change vs the PLAIN file of the same members, C2 ('FD8', 'FDP'): share of feeds with the same top-1 | mean per-feed Spearman   (all / public-like / private-like)
 MRULE (a .2 | .4 by M>=50)                      : same top-1 0.9536 / 0.9497 / 0.9553 | Spearman 0.9993 / 0.9993 / 0.9993 | feeds with any order change 0.610
 M-rule .2 | .5                                  : same top-1 0.9330 / 0.9272 / 0.9355 | Spearman 0.9986 / 0.9986 / 0.9986 | feeds with any order change 0.610
 h3-TO (M-rule in the train-only branch only)    : same top-1 0.9710 / 0.9672 / 0.9726 | Spearman 0.9997 / 0.9997 / 0.9997 | feeds with any order change 0.610
 h4a (.2,.3,.4 by M 30/100)                      : same top-1 0.9573 / 0.9516 / 0.9597 | Spearman 0.9993 / 0.9993 / 0.9993 | feeds with any order change 0.745
 F(Hq) best on C1 (a .4; c .7 if H<190 else .5)  : same top-1 0.9309 / 0.9267 / 0.9327 | Spearman 0.9973 / 0.9975 / 0.9972 | feeds with any order change 0.908
 CONST a .4 c .5                                 : same top-1 0.9350 / 0.9312 / 0.9366 | Spearman 0.9976 / 0.9977 / 0.9976 | feeds with any order change 0.903
 CONST a .4 c .7 (A best on C1)                  : same top-1 0.9265 / 0.9219 / 0.9284 | Spearman 0.9969 / 0.9970 / 0.9968 | feeds with any order change 0.915
```

- M and H gates have the same cell shares on val, public-like and private-like test (M >= 50: 0.591 / 0.613 / 0.608). Cold-share gates do not (CS >= 0.45: 0.084 / 0.215 / 0.327): a cold-share rule fitted on val would act on 3-4 times more feeds on private than it was fitted on. U0 cells hold 4.1% of test feeds (below the declared 5%).
- The M-rule file keeps the top-1 item of the PLAIN file on 95.6% of test feeds (92.8% of the long ones; the 39% short feeds are untouched), mean per-feed Spearman 0.9993; the same on public-like and private-like feeds.

## 7. Descriptive dose tables (addenda 3-5; fixed rules, in-sample, no selection claimed) [M]

Constant w_pop at w_ft 0.5 by M cell, minus 0.2; first line of each block = popularity-light pool alone minus base pool alone:

```
cell sizes: all 14423, M<30 3883, M30-49 2010, M50-99 3182, M>=100 5348, M<50 5893, M>=50 8530
=== C1 ('FD16', 'FDP'): pool-only finals (0.5 FT + 0.5 TO): popularity-light pool minus base pool by cell: all -0.00024+-0.00077 | M<30 -0.00235+-0.00176 | M30-49 -0.00230+-0.00224 | M50-99 +0.00088+-0.00168 | M>=100 +0.00140+-0.00098 | M<50 -0.00233+-0.00139 | M>=50 +0.00121+-0.00088
   const c at a = .2 (minus c = .5), all feeds: c 0.3 -0.00015+-0.00026 | c 0.4 -0.00008+-0.00020 | c 0.5 +0.00000+-0.00000 | c 0.6 -0.00011+-0.00016 | c 0.7 -0.00038+-0.00024 | c 0.8 -0.00037+-0.00030
a (c = .5) minus a = .2  |                all |               M<30 |             M30-49 |             M50-99 |             M>=100 |               M<50 |              M>=50
a = 0.0                  | -0.00085+-0.00031 | -0.00019+-0.00076 | -0.00080+-0.00085 | -0.00106+-0.00061 | -0.00121+-0.00039 | -0.00040+-0.00058 | -0.00115+-0.00033
a = 0.1                  | -0.00007+-0.00021 | +0.00071+-0.00055 | -0.00007+-0.00055 | -0.00033+-0.00040 | -0.00049+-0.00026 | +0.00044+-0.00041 | -0.00043+-0.00022
a = 0.2                  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000
a = 0.3                  | +0.00015+-0.00022 | -0.00058+-0.00053 | +0.00070+-0.00054 | +0.00076+-0.00044 | +0.00012+-0.00028 | -0.00014+-0.00040 | +0.00036+-0.00024
a = 0.4                  | +0.00048+-0.00032 | +0.00004+-0.00074 | +0.00146+-0.00089 | +0.00045+-0.00071 | +0.00044+-0.00039 | +0.00053+-0.00058 | +0.00044+-0.00036
a = 0.5                  | +0.00011+-0.00039 | -0.00098+-0.00091 | +0.00040+-0.00113 | +0.00071+-0.00087 | +0.00044+-0.00047 | -0.00051+-0.00071 | +0.00054+-0.00044
a = 0.6                  | -0.00010+-0.00045 | -0.00082+-0.00103 | -0.00125+-0.00136 | +0.00055+-0.00097 | +0.00047+-0.00055 | -0.00097+-0.00082 | +0.00050+-0.00050
=== C2 ('FD8', 'FDP'): pool-only finals (0.5 FT + 0.5 TO): popularity-light pool minus base pool by cell: all +0.00015+-0.00078 | M<30 -0.00149+-0.00182 | M30-49 -0.00158+-0.00229 | M50-99 +0.00121+-0.00164 | M>=100 +0.00137+-0.00100 | M<50 -0.00152+-0.00143 | M>=50 +0.00131+-0.00088
   const c at a = .2 (minus c = .5), all feeds: c 0.3 -0.00024+-0.00023 | c 0.4 -0.00017+-0.00017 | c 0.5 +0.00000+-0.00000 | c 0.6 -0.00010+-0.00017 | c 0.7 +0.00004+-0.00025 | c 0.8 -0.00008+-0.00030
a (c = .5) minus a = .2  |                all |               M<30 |             M30-49 |             M50-99 |             M>=100 |               M<50 |              M>=50
a = 0.0                  | -0.00092+-0.00029 | -0.00080+-0.00072 | -0.00151+-0.00090 | -0.00062+-0.00059 | -0.00096+-0.00034 | -0.00104+-0.00056 | -0.00084+-0.00031
a = 0.1                  | -0.00039+-0.00021 | -0.00026+-0.00048 | -0.00116+-0.00074 | +0.00002+-0.00041 | -0.00044+-0.00024 | -0.00056+-0.00041 | -0.00027+-0.00021
a = 0.2                  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000
a = 0.3                  | +0.00022+-0.00022 | -0.00032+-0.00054 | +0.00029+-0.00052 | +0.00113+-0.00045 | +0.00005+-0.00029 | -0.00011+-0.00040 | +0.00045+-0.00025
a = 0.4                  | +0.00050+-0.00032 | -0.00048+-0.00079 | +0.00089+-0.00082 | +0.00150+-0.00067 | +0.00048+-0.00040 | -0.00001+-0.00059 | +0.00086+-0.00035
a = 0.5                  | +0.00033+-0.00039 | -0.00024+-0.00093 | -0.00002+-0.00112 | +0.00100+-0.00079 | +0.00048+-0.00049 | -0.00017+-0.00072 | +0.00067+-0.00043
a = 0.6                  | +0.00004+-0.00045 | -0.00060+-0.00106 | -0.00103+-0.00133 | +0.00095+-0.00091 | +0.00036+-0.00057 | -0.00074+-0.00084 | +0.00058+-0.00049
=== C3 ('W16', 'P1w'): pool-only finals (0.5 FT + 0.5 TO): popularity-light pool minus base pool by cell: all -0.00082+-0.00078 | M<30 -0.00420+-0.00176 | M30-49 -0.00412+-0.00215 | M50-99 +0.00271+-0.00162 | M>=100 +0.00077+-0.00107 | M<50 -0.00417+-0.00137 | M>=50 +0.00150+-0.00090
   const c at a = .2 (minus c = .5), all feeds: c 0.3 +0.00004+-0.00026 | c 0.4 -0.00018+-0.00018 | c 0.5 +0.00000+-0.00000 | c 0.6 +0.00011+-0.00017 | c 0.7 -0.00015+-0.00024 | c 0.8 -0.00033+-0.00030
a (c = .5) minus a = .2  |                all |               M<30 |             M30-49 |             M50-99 |             M>=100 |               M<50 |              M>=50
a = 0.0                  | -0.00056+-0.00032 | -0.00037+-0.00078 | -0.00002+-0.00080 | -0.00074+-0.00064 | -0.00080+-0.00042 | -0.00025+-0.00058 | -0.00078+-0.00036
a = 0.1                  | -0.00044+-0.00022 | -0.00048+-0.00051 | -0.00015+-0.00054 | -0.00027+-0.00043 | -0.00064+-0.00034 | -0.00036+-0.00038 | -0.00050+-0.00027
a = 0.2                  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000
a = 0.3                  | +0.00022+-0.00020 | -0.00093+-0.00044 | -0.00036+-0.00063 | +0.00114+-0.00043 | +0.00072+-0.00029 | -0.00074+-0.00036 | +0.00088+-0.00024
a = 0.4                  | +0.00022+-0.00030 | -0.00104+-0.00064 | -0.00148+-0.00087 | +0.00162+-0.00063 | +0.00094+-0.00041 | -0.00119+-0.00051 | +0.00119+-0.00035
a = 0.5                  | +0.00016+-0.00037 | -0.00138+-0.00086 | -0.00196+-0.00106 | +0.00196+-0.00077 | +0.00099+-0.00051 | -0.00157+-0.00067 | +0.00135+-0.00043
a = 0.6                  | -0.00014+-0.00045 | -0.00182+-0.00104 | -0.00227+-0.00131 | +0.00204+-0.00091 | +0.00058+-0.00059 | -0.00198+-0.00082 | +0.00112+-0.00050
=== C4 ('W8', 'P1w'): pool-only finals (0.5 FT + 0.5 TO): popularity-light pool minus base pool by cell: all -0.00089+-0.00078 | M<30 -0.00420+-0.00173 | M30-49 -0.00363+-0.00223 | M50-99 +0.00212+-0.00166 | M>=100 +0.00075+-0.00106 | M<50 -0.00400+-0.00137 | M>=50 +0.00126+-0.00091
   const c at a = .2 (minus c = .5), all feeds: c 0.3 +0.00000+-0.00026 | c 0.4 -0.00005+-0.00018 | c 0.5 +0.00000+-0.00000 | c 0.6 +0.00020+-0.00017 | c 0.7 -0.00010+-0.00024 | c 0.8 -0.00029+-0.00030
a (c = .5) minus a = .2  |                all |               M<30 |             M30-49 |             M50-99 |             M>=100 |               M<50 |              M>=50
a = 0.0                  | -0.00053+-0.00031 | +0.00041+-0.00072 | -0.00078+-0.00095 | -0.00167+-0.00073 | -0.00045+-0.00035 | +0.00001+-0.00057 | -0.00090+-0.00035
a = 0.1                  | -0.00021+-0.00022 | +0.00058+-0.00052 | -0.00085+-0.00065 | -0.00114+-0.00050 | +0.00000+-0.00027 | +0.00010+-0.00041 | -0.00042+-0.00025
a = 0.2                  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000
a = 0.3                  | +0.00005+-0.00021 | -0.00018+-0.00053 | -0.00069+-0.00056 | +0.00002+-0.00046 | +0.00051+-0.00026 | -0.00035+-0.00040 | +0.00032+-0.00024
a = 0.4                  | -0.00007+-0.00029 | -0.00057+-0.00070 | -0.00247+-0.00087 | +0.00046+-0.00060 | +0.00088+-0.00037 | -0.00122+-0.00055 | +0.00072+-0.00032
a = 0.5                  | -0.00025+-0.00037 | -0.00133+-0.00088 | -0.00297+-0.00106 | +0.00094+-0.00080 | +0.00085+-0.00048 | -0.00189+-0.00068 | +0.00088+-0.00042
a = 0.6                  | -0.00055+-0.00044 | -0.00183+-0.00105 | -0.00330+-0.00125 | +0.00099+-0.00092 | +0.00049+-0.00056 | -0.00234+-0.00081 | +0.00068+-0.00049
```

- M >= 50: 0.4 minus 0.2 = +0.00044 / +0.00086 / +0.00119 / +0.00072 per long feed; the top is flat between 0.4 and 0.5 (0.5: +0.00054 / +0.00067 / +0.00135 / +0.00088).
- M < 50: 0.4 minus 0.2 = +0.00053 / -0.00001 on the full-data members, -0.00119 / -0.00122 on the twins. Post hoc contrast (largest of 6 looked at): full-data minus twins +0.00172 +- 0.00067 (16-base), +0.00121 +- 0.00072 (8-base). Decomposition (addendum 5): both the full-data base pool and the full-data popularity-light pool move it by about +0.0007..+0.0010 each on the 16-base (1.1-1.6 SE each; +0.0004..+0.0008 on the 8-base): "not attributable" to one pool; 7 of the 8 crossings are at or below zero on short feeds, mean -0.0005. So 0.2 on short feeds stands; "0.4 everywhere" is not supported.

```
C1  FD16 base + FDP (full-data pop-light)     : a .4 - .2  M<50 +0.00053+-0.00058  M>=50 +0.00044+-0.00036 all +0.00048+-0.00032 | a .3 - .2  M<50 -0.00014+-0.00040 | plain 0.40271
X1  FD16 base + P1w (held-out pop-light)      : a .4 - .2  M<50 -0.00018+-0.00054  M>=50 +0.00054+-0.00037 all +0.00025+-0.00031 | a .3 - .2  M<50 +0.00002+-0.00039 | plain 0.40234
X2  W16 base + FDP (full-data pop-light)      : a .4 - .2  M<50 -0.00021+-0.00057  M>=50 +0.00104+-0.00036 all +0.00053+-0.00032 | a .3 - .2  M<50 +0.00032+-0.00042 | plain 0.40201
C3  W16 base + P1w (held-out pop-light)       : a .4 - .2  M<50 -0.00119+-0.00051  M>=50 +0.00119+-0.00035 all +0.00022+-0.00030 | a .3 - .2  M<50 -0.00074+-0.00036 | plain 0.40214
paired on M<50: pop-light pool effect (FDP minus P1w): with FD16 base +0.00071+-0.00062, with W16 base +0.00098+-0.00059 | base pool effect (FD16 minus W16): with FDP +0.00074+-0.00063, with P1w +0.00101+-0.00062
C2  FD8 base + FDP                            : a .4 - .2  M<50 -0.00001+-0.00059  M>=50 +0.00086+-0.00035 all +0.00050+-0.00032 | a .3 - .2  M<50 -0.00011+-0.00040 | plain 0.40240
X3  FD8 base + P1w                            : a .4 - .2  M<50 -0.00067+-0.00057  M>=50 +0.00076+-0.00038 all +0.00018+-0.00032 | a .3 - .2  M<50 -0.00009+-0.00039 | plain 0.40223
X4  W8 base + FDP                             : a .4 - .2  M<50 -0.00082+-0.00059  M>=50 +0.00105+-0.00036 all +0.00029+-0.00032 | a .3 - .2  M<50 -0.00062+-0.00044 | plain 0.40210
C4  W8 base + P1w                             : a .4 - .2  M<50 -0.00122+-0.00055  M>=50 +0.00072+-0.00032 all -0.00007+-0.00029 | a .3 - .2  M<50 -0.00035+-0.00040 | plain 0.40218
paired on M<50 (8-base): pop-light pool effect: with FD8 base +0.00066+-0.00063, with W8 base +0.00040+-0.00060 | base pool effect: with FDP +0.00081+-0.00069, with P1w +0.00055+-0.00070
```
```
a = 0.3 minus 0.2, C1 (full data) minus C3 (held-out twins): M<50 +0.00060+-0.00050 (z +1.19) | M>=50 -0.00052+-0.00031 (z -1.68)
a = 0.3 minus 0.2, C2 (full data) minus C4 (held-out twins): M<50 +0.00024+-0.00054 (z +0.44) | M>=50 +0.00013+-0.00031 (z +0.42)
a = 0.4 minus 0.2, C1 (full data) minus C3 (held-out twins): M<50 +0.00172+-0.00067 (z +2.55) | M>=50 -0.00075+-0.00042 (z -1.79)
a = 0.4 minus 0.2, C2 (full data) minus C4 (held-out twins): M<50 +0.00121+-0.00072 (z +1.69) | M>=50 +0.00014+-0.00039 (z +0.35)
a = 0.5 minus 0.2, C1 (full data) minus C3 (held-out twins): M<50 +0.00107+-0.00081 (z +1.32) | M>=50 -0.00081+-0.00048 (z -1.68)
a = 0.5 minus 0.2, C2 (full data) minus C4 (held-out twins): M<50 +0.00173+-0.00083 (z +2.08) | M>=50 -0.00021+-0.00048 (z -0.43)
```

- A clean test of that lead exists once the 8 new wide full-data students (s955-962) are fine-tuned: on FD16 + the NEW students only, read "0.4 minus 0.2 on M < 50". If it is >= +1 SE there, a short-feed weight of 0.3-0.4 for full-data files becomes a candidate; otherwise the lead is closed. One difference between the two member sets that could be systematic [I]: held-out members use a checkpoint selected on the holdout with M >= 50, full-data members the last step.

Constant w_ft (c) minus 0.5, unweighted and under pmix weights (c = 0 is the train-only branch alone, 1 the fine-tuned branch alone):

```
=== C1 ('FD16', 'FDP') | a = .2 | level at c = .5: val 0.40271 pmix-public 0.39998 pmix-private 0.40562
c    | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n 1213) | rest
0.0  | -0.00058+-0.00041 | -0.00070+-0.00051 | -0.00061+-0.00062 | -0.00007+-0.00152      | -0.00063+-0.00042
0.5  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000      | +0.00000+-0.00000
0.6  | -0.00011+-0.00016 | -0.00009+-0.00020 | -0.00007+-0.00024 | +0.00023+-0.00049      | -0.00014+-0.00017
0.7  | -0.00038+-0.00024 | -0.00010+-0.00029 | -0.00002+-0.00036 | +0.00023+-0.00088      | -0.00043+-0.00025
1.0  | -0.00081+-0.00040 | -0.00082+-0.00049 | -0.00075+-0.00059 | +0.00058+-0.00154      | -0.00094+-0.00041
=== C1 ('FD16', 'FDP') | M-rule weights | level at c = .5: val 0.40298 pmix-public 0.40024 pmix-private 0.40600
c    | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n 1213) | rest
0.0  | -0.00068+-0.00040 | -0.00088+-0.00050 | -0.00097+-0.00062 | -0.00112+-0.00158      | -0.00064+-0.00041
0.5  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000      | +0.00000+-0.00000
0.6  | -0.00005+-0.00018 | +0.00013+-0.00022 | +0.00016+-0.00026 | +0.00044+-0.00040      | -0.00010+-0.00019
0.7  | +0.00007+-0.00026 | +0.00035+-0.00031 | +0.00035+-0.00037 | +0.00003+-0.00077      | +0.00007+-0.00027
1.0  | -0.00070+-0.00040 | -0.00037+-0.00049 | -0.00027+-0.00059 | +0.00071+-0.00149      | -0.00083+-0.00042
=== C2 ('FD8', 'FDP') | a = .2 | level at c = .5: val 0.40240 pmix-public 0.39963 pmix-private 0.40531
c    | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n 1213) | rest
0.0  | -0.00039+-0.00039 | -0.00034+-0.00048 | -0.00022+-0.00059 | +0.00090+-0.00141      | -0.00051+-0.00041
0.5  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000      | +0.00000+-0.00000
0.6  | -0.00010+-0.00017 | -0.00010+-0.00021 | -0.00013+-0.00025 | -0.00017+-0.00063      | -0.00009+-0.00017
0.7  | +0.00004+-0.00025 | +0.00029+-0.00032 | +0.00043+-0.00039 | +0.00166+-0.00110      | -0.00011+-0.00025
1.0  | -0.00056+-0.00041 | -0.00031+-0.00052 | -0.00036+-0.00063 | -0.00082+-0.00157      | -0.00053+-0.00043
=== C2 ('FD8', 'FDP') | M-rule weights | level at c = .5: val 0.40290 pmix-public 0.40022 pmix-private 0.40603
c    | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n 1213) | rest
0.0  | -0.00079+-0.00040 | -0.00087+-0.00049 | -0.00099+-0.00060 | -0.00053+-0.00142      | -0.00081+-0.00041
0.5  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000      | +0.00000+-0.00000
0.6  | -0.00018+-0.00016 | -0.00021+-0.00020 | -0.00027+-0.00025 | -0.00021+-0.00068      | -0.00018+-0.00017
0.7  | +0.00006+-0.00025 | +0.00018+-0.00031 | +0.00018+-0.00039 | +0.00121+-0.00111      | -0.00004+-0.00025
1.0  | -0.00086+-0.00039 | -0.00051+-0.00049 | -0.00055+-0.00060 | -0.00055+-0.00153      | -0.00089+-0.00041
=== C3 ('W16', 'P1w') | a = .2 | level at c = .5: val 0.40214 pmix-public 0.39925 pmix-private 0.40490
c    | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n 1213) | rest
0.0  | -0.00024+-0.00042 | -0.00063+-0.00051 | -0.00083+-0.00063 | -0.00110+-0.00159      | -0.00016+-0.00043
0.5  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000      | +0.00000+-0.00000
0.6  | +0.00011+-0.00017 | +0.00011+-0.00021 | +0.00010+-0.00024 | +0.00013+-0.00034      | +0.00011+-0.00018
0.7  | -0.00015+-0.00024 | -0.00018+-0.00030 | -0.00018+-0.00036 | +0.00028+-0.00065      | -0.00019+-0.00026
1.0  | -0.00075+-0.00040 | -0.00088+-0.00051 | -0.00102+-0.00062 | -0.00259+-0.00161      | -0.00058+-0.00041
=== C3 ('W16', 'P1w') | M-rule weights | level at c = .5: val 0.40285 pmix-public 0.40006 pmix-private 0.40583
c    | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n 1213) | rest
0.0  | -0.00048+-0.00043 | -0.00065+-0.00053 | -0.00086+-0.00065 | -0.00134+-0.00160      | -0.00040+-0.00044
0.5  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000      | +0.00000+-0.00000
0.6  | -0.00018+-0.00016 | -0.00028+-0.00019 | -0.00035+-0.00024 | -0.00044+-0.00050      | -0.00016+-0.00017
0.7  | -0.00052+-0.00023 | -0.00053+-0.00029 | -0.00054+-0.00037 | -0.00010+-0.00084      | -0.00056+-0.00024
1.0  | -0.00099+-0.00039 | -0.00110+-0.00049 | -0.00122+-0.00060 | -0.00270+-0.00159      | -0.00083+-0.00040
=== C4 ('W8', 'P1w') | a = .2 | level at c = .5: val 0.40218 pmix-public 0.39907 pmix-private 0.40470
c    | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n 1213) | rest
0.0  | -0.00036+-0.00042 | -0.00045+-0.00051 | -0.00056+-0.00062 | -0.00013+-0.00162      | -0.00038+-0.00043
0.5  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000      | +0.00000+-0.00000
0.6  | +0.00020+-0.00017 | +0.00019+-0.00021 | +0.00010+-0.00026 | -0.00063+-0.00062      | +0.00028+-0.00017
0.7  | -0.00010+-0.00024 | +0.00011+-0.00032 | +0.00012+-0.00040 | +0.00005+-0.00098      | -0.00011+-0.00025
1.0  | -0.00035+-0.00039 | -0.00001+-0.00051 | +0.00015+-0.00063 | +0.00157+-0.00161      | -0.00053+-0.00040
=== C4 ('W8', 'P1w') | M-rule weights | level at c = .5: val 0.40261 pmix-public 0.39983 pmix-private 0.40562
c    | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n 1213) | rest
0.0  | -0.00039+-0.00042 | -0.00070+-0.00052 | -0.00102+-0.00064 | -0.00126+-0.00165      | -0.00031+-0.00043
0.5  | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000 | +0.00000+-0.00000      | +0.00000+-0.00000
0.6  | +0.00009+-0.00018 | +0.00019+-0.00022 | +0.00016+-0.00025 | -0.00016+-0.00049      | +0.00011+-0.00019
0.7  | -0.00028+-0.00024 | -0.00016+-0.00029 | -0.00014+-0.00035 | +0.00013+-0.00079      | -0.00032+-0.00025
1.0  | -0.00054+-0.00040 | -0.00045+-0.00050 | -0.00039+-0.00061 | +0.00088+-0.00155      | -0.00067+-0.00041
```

- No constant 0.6 / 0.7 reads +1 SE at the private mix on C1 and stays positive on all four: w_ft = 0.5 stands on val at every mix. Val cannot say more: the fine-tuned branch is read out-of-fold on val, and its public value (+0.0005 [D]) is above its val value.

## 8. Candidates for tomorrow's public slots

**None.** Nothing beats the M-rule family nested by more than its placebo noise (family-wise p for "some family beats C(M50) across two member sets": 0.94).

| | definition | nested gain over PLAIN, C1 | over the M-rule, C1 | placebo p | replication | P(gains on public) |
|---|---|---|---|---|---|---|
| standing rule (already in the owner's slots) | w_pop 0.2 if M < 50 else 0.4, both branches; w_ft 0.5 | family refit +0.00018 +- 0.00030; fixed rule +0.00026 +- 0.00021 | 0 (reference); fixed minus refit +0.00008 +- 0.00020 | 0.19 on C1 alone; 0.040 for the mean of two member sets (+0.00036 +- 0.00022) | nested +0.00021 / +0.00055 / +0.00028 on C2 / C3 / C4; other models + other users +0.00027 / +0.00036 / +0.00014 / +0.00008; over 8 compositions nested +0.00028 +- 0.00017, fixed +0.00041 +- 0.00014, 7 of 8 positive each | against the PLAIN file of the same members: 0.70 for the FD8 pair, 0.60 for the FD16 pair (the FD16 base is where the rule is weakest: +0.00026, -0.00010 fixed; +0.00018, -0.00011 nested); expected public delta about +0.0003 / +0.0002, private about a third more [I] |
| NOT a candidate (shown because it tops the C1 table) | w_pop 0.4 everywhere; w_ft 0.7 if H < 190 else 0.5 | +0.00076 +- 0.00032 (in-sample +0.00076) | +0.00050 +- 0.00027 over the fixed M-rule; +0.00058 over the family (p 0.005) | 0.010; family-wise 0.030 | fails: C3 -0.00019, C4 -0.00034; twins -> C1 -0.00005; two-set mean +0.00029 (p 0.059), below the fixed M-rule by 0.00020 +- 0.00019 | about 0.35-0.4 against the M-rule file [I] |

Build recipe of the standing rule (equal to build_r4.py's `@m0.2_0.4` path; `lib.ref_final` is this function and reproduces the levels on record at all 5 decimals, section 1):

```python
import numpy as np
from stack_val import feed_rank_z          # src/
def gated_mix(bF, bT, pF, pT, off, M, a_ft, a_to, c):
    """bF, bT, pF, pT: base / popularity-light pool arrays (member-equal mean of per-feed z; fine-tuned, train-only), rows of the candidate table.
    a_ft, a_to, c: per-FEED arrays of len(M) (popularity-light weight in each branch, weight of the fine-tuned branch)."""
    Z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
    r = lambda v: np.repeat(np.asarray(v, np.float32), M)
    brF = (1 - r(a_ft)) * Z(bF) + r(a_ft) * Z(pF)
    brT = (1 - r(a_to)) * Z(bT) + r(a_to) * Z(pT)
    return r(c) * Z(brF) + (1 - r(c)) * Z(brT)
def m_rule(bF, bT, pF, pT, off, M):
    a = np.where(M >= 50, 0.4, 0.2)
    return gated_mix(bF, bT, pF, pT, off, M, a, a, np.full(len(M), 0.5))
```

What I would do with the result [I]:
- Keep the weights 0.2 / 0.4 as the lead declared at 12:55Z, also when the popularity-light pool grows (h2: 0.5 is not better with 8 members; the top is flat 0.4-0.5 with 4).
- Between the two paired public tests, the FD8 pair is the better-founded one; on the FD16 base the rule is not shown on val (fixed +0.00026 / -0.00010, nested +0.00018 / -0.00011). If only one M-rule file is carried forward on FD16, its val advantage over the plain FD16 file should be treated as about +0.0002 +- 0.0003 (between the FD16-specific reads and the 8-composition mean), and the public pair decides.
- In the reading rule of the public pair, replace the in-sample val leg (+0.00051 +- 0.00021 for the FD8 pair) by the nested number (+0.00036 +- 0.00022). Pooled +2 SE then needs D_pub >= about +0.0003 (paired public SE 0.00027 [D]); with the in-sample leg it needs only D_pub >= 0.
- Do not read a w_ft or cold-share rule from val: val has 8% cold feeds against 21-33% on test and reads the fine-tuned branch out-of-fold. If the question matters, it needs a public pair, and the prior from val is "no effect" (h1).

## 9. Rule compliance and breaches

- CPU only; at most 2 heavy processes (12 threads each); own jobs only, none stopped; nothing written outside `nn_x/w12/mix/rulesA/` and this report; no network; no submission file. `nvidia-smi` was queried once (read-only).
- Breach 1: one foreground `sleep 20` (single, after launching the first two background jobs).
- Breach 2 (minor): short reads capped at 2 threads (hyp.py, dose.py, criteria.py, cdose.py, t2_read.py; each under 20 s) ran next to two heavy jobs.
- Declaration errors, corrected in addenda before use: the count "93 eligible families" is 95; the stop rule of block 5 used the SE over draws (not a valid SE for that mean); it tripped, the leak check of addendum 2 was run and passed.
- Post hoc elements (declared in addenda before their numbers, after earlier numbers were seen): cross-model read, two-member-set statistic T2 with matched placebo, w_ft curve under pmix, dose tables, the full-data-versus-twins contrast. The verdict of section 0 rests on the criteria of the original declaration; T2 only sizes the M-rule family.
- Val labels were used only through user-grouped nested procedures for every fitted rule; every fitted number is quoted nested with its in-sample number next to it. Fixed rules (hypotheses, dose tables) are read on full val and are marked in-sample / descriptive where they were chosen on these feeds. Test rows were used label-free only.
- Not done: PL-G on C2-C4; placebo for the 8-base compositions and for C5-C8; h2 with 8 WIDE full-data students (s955-958 not finished at the time of the last read, 13:37Z).

## Appendix: DECLARATION.txt (verbatim)

```
=== DECLARATION rulesA (searcher A, declared rule families) ; written 2026-11-15T13:16:02Z BEFORE any labelled number of this study ===
Labelled numbers seen so far by this agent: only numbers on record in reports/CURRENT_STATUS.md sections 1-4 and nn_x/w5/logs/E_kd_predeclared.txt (blocks 09:27Z-12:55Z) [D]. Label-free looks done before this
declaration: quantiles of M, H, cold share, user0 share, time stamps on val / public-like / private-like test (to fix thresholds).

BLOCK 0. REPRODUCTION (step 1; bar fixed here)
 Own loader (member files -> per-feed z -> pools -> branches -> final), project functions feed_rank_z / ndcg_per_feed for the reference path and an own numba kernel for the grid path.
 PASS iff every one of these final-val levels, rounded to 5 decimals, equals the number on record: FD16P20 0.40271, FD16P20@m0.2_0.4 0.40298, FD8P20 0.40240, FD8P20@m0.2_0.4 0.40290, W16P20w 0.40214,
 W16P20w@m0.2_0.4 0.40285; and rule-minus-plain deltas FD16 +0.00026+-0.00021, W16P20w +0.00071+-0.00021, W8P20w +0.00043+-0.00019 (to 1e-5).
 Kernel bar: own numba kernel (float64, pessimistic ties) vs the reference path: max abs difference of the mean final val over the checked rules <= 1e-5 AND mean abs per-feed difference <= 1e-5. If FAIL: stop and fix before any search.

BLOCK 1. OBJECTS
 Compositions (base pool + popularity-light pool; all 4 pools = member-equal mean of per-feed z of the member files, as build_r4.py):
   C1 PRIMARY  FD16 + 4 wide full-data popularity-light (FDP s951-954)            = FD16P20
   C2          FD8  + 4 wide full-data popularity-light                             = FD8P20
   C3 (replication on other trained models, SAME val feeds) W16 + 4 wide held-out (P1w) = W16P20w
   C4 (replication, same val feeds)                         W8  + 4 wide held-out      = W8P20w
 Score of a feed under weights (a_ft, a_to, c):  final = c * z( (1-a_ft) z(baseFT) + a_ft z(popFT) ) + (1-c) * z( (1-a_to) z(baseTO) + a_to z(popTO) ),  z = per-feed standardisation; per-feed NDCG as src/metrics.py.
 Weight grids: a (w_pop) in {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}; c (w_ft) in {0.3, 0.4, 0.5, 0.6, 0.7, 0.8}.
 Baselines: PLAIN = (a_ft = a_to = 0.2, c = 0.5) on every feed. MRULE = a_ft = a_to = 0.2 if M < 50 else 0.4, c = 0.5 (fixed rule, declared by the lead 09:27Z; in-sample on these val feeds).

BLOCK 2. GATES (label-free feed properties; thresholds fixed HERE, never tuned to labels)
 Variables: M (candidates), H (history length), U0 (user_id == 0), CL = "luo cold share" = share of candidates with it_hist_all_luo == 0 (nn/feat_cache; the task's definition), CS = "segment cold share" = share of candidates whose
 item is in no train candidate list / history (eda/errors/segments feed_table / test_segments; the pmix variable; the variable behind the on-record statement "FT value sits in feeds with cold share >= 45%"), DAY (val: first vs second val day).
 Quantiles are those of the POOLED val + test feeds (label-free): M 39 / 70 / 117 (1/3, 1/2, 2/3), CL 0.3737 / 0.4225 / 0.4744, H 141 / 190 / 241.
 Two-cell gates (cell 1 = condition true):
   M50  : M >= 50            (the M-rule's cut; on record, originally label-prompted)
   Mq   : M >= 70            (pooled median)
   CS45 : CS >= 0.45         (pmix bin edge; on record)
   CLq  : CL >= 0.4225       (pooled median)
   Hq   : H >= 190           (pooled median)
   U0   : user_id == 0
   DAY  : second val day     (DIAGNOSTIC ONLY: test lies after both val days, so the gate does not cover the same range on test; never a candidate; reported as a transfer warning light)
 Three-cell gates:
   M3q  : M < 39 | 39 <= M < 117 | M >= 117   (pooled terciles)
   M3r  : M < 30 | 30 <= M < 100 | M >= 100   (the cut of hypothesis h4)
   CL3q : CL < 0.3737 | .. | CL >= 0.4744
   H3q  : H < 141 | 141 <= H < 241 | H >= 241

BLOCK 3. RULE FAMILIES (piecewise constant; every family has <= 4 free weight parameters; "a" = w_pop used in both branches unless stated)
   A   CONST2        a, c constant                                             2 par   42 rules      1 family
   B   CONST3        a_ft, a_to, c constant                                    3 par   294           1
   C   POP1(g)       a by two-cell gate g; c = 0.5                             2 par   49            7 (6 eligible + DAY)        [C(M50) = the M-rule's family]
   D   POP1F(g)      a by g; c constant free                                   3 par   294           7
   E   FT1(g)        c by g; a = 0.2                                           2 par   36            7
   F   FT1P(g)       c by g; a constant free                                   3 par   252           7
   G   POP3(t)       a by three-cell gate t; c = 0.5                           3 par   343           4
   H   POP3F(t)      a by t; c constant free                                   4 par   2058          4
   I   BR(g)         a_ft by g and a_to by g (separate per branch); c = 0.5    4 par   2401          7
   J   POPxFT(g,g')  a by g, c by g' (g, g' eligible two-cell gates, may be equal) 4 par 1764        36
   K   POP2D(g,g')   a on the 2 x 2 cells of two different eligible gates; c = 0.5  4 par 2401       15
   L   FT3(t)        c by three-cell gate t; a = 0.2                           3 par   216           4
 Total 100 families (93 eligible: every family with DAY is diagnostic only), about 131,000 rules. PLAIN is a member of A, B, C, D, E, F, G, H, I, J, K, L (every family). MRULE is a member of C(M50), D(M50), I(M50), J(M50, .), K(M50, .).

BLOCK 4. NESTED READ (step 3)
 Folds: K = 5 user-grouped folds, 5 fold seeds (seeds 11, 22, 33, 44, 55); a user's feeds never straddle folds; each user_id 0 feed is its own group. Users are assigned by a seeded permutation of the unique groups.
 Variant A (per family): in each training part (4 folds) the rule with the highest mean NDCG of the training feeds (ties: first in enumeration order, which lists the plain-like values in natural grid order) is applied to the held-out
   fold. Nested per-feed NDCG of a seed = concatenation over the 5 held-out folds; the reported nested vector is the per-feed mean over the 5 seeds.
 Variant B (best family; double nested): inside each training part, each family is scored by inner cross-validation over its 4 folds (rule fitted on 3 folds, read on the 4th; inner score = mean over the training feeds); the family
   with the highest inner score is taken, its best rule on the whole training part is applied to the held-out fold. B-all = over the 93 eligible families; B-3 = over the eligible families with <= 3 parameters (A-G, L).
 Variant G (greedy, for the record of the selection bias): the single rule with the highest training mean over ALL eligible families (the richest families win by construction).
 Numbers reported for each family / variant and each composition: in-sample gain (best rule on all val minus PLAIN), nested gain minus PLAIN, nested minus MRULE (fixed), nested minus nested C(M50) (the fair read of "better than
   the M-rule family"), each with the paired per-feed SE (sd of the per-feed difference / sqrt(14423); selection variance is NOT in this SE: the placebo gives it), plus the range over the 5 fold seeds.
 Multiplicity is disclosed as: 100 families, about 131,000 rules, 3 selection variants, 4 compositions, 2 baselines.

BLOCK 5. PLACEBO (step 4; measures the selection bias and the noise of the whole search)
 PL-A (the task's placebo): draw d = 1..200 on the FD16 members: 4 random members of the 16 wide students play the "popularity-light pool", the other 12 the base pool; a random half of the base (6) and of the pseudo pool (2)
   swap their fine-tuned / train-only files (so "branch FT" and "branch TO" are both half fine-tuned, half train-only). The IDENTICAL search (families, grids, folds, variants) is run; recorded per draw and family / variant:
   in-sample gain and nested gain over the placebo's own PLAIN (0.2, 0.5), nested minus nested C(M50), nested minus nested A.
   Same with the held-out W16 members, 100 draws (run if time allows; if not, stated as not run).
 PL-G (gate permutation; real pools of C1): the vector of feed properties (all gates jointly) is permuted across feeds (200 permutations), predictions and labels untouched; the identical search; recorded as PL-A.
   PL-G is the null "the gate carries no information" at the real pools: its reference quantity is nested(family) minus nested(A).
 Placebo-calibrated p-value of an observed nested quantity q: p = (1 + #{draws with placebo q_d >= q}) / (1 + N). Reported: p_A (PL-A, gain over PLAIN), p_A,M (PL-A, family minus C(M50)), p_G (PL-G, family minus A).
 Expected under a sound procedure: placebo nested gains centred at or below 0 (a positive mean beyond 2 SE of the mean = a leak in my procedure: STOP and find it); placebo in-sample gains positive = the optimism.

BLOCK 6. CANDIDATE CRITERIA (fixed here; judged on the PRIMARY composition C1 unless stated)
 A family (variant A) or a selection variant (B-all, B-3) yields a CANDIDATE iff ALL of:
   c1  nested gain over PLAIN >= +2 paired SE and p_A <= 0.05;
   c2  nested minus nested C(M50) > the 95th percentile of the same difference over the PL-A draws (p_A,M <= 0.05) AND nested minus fixed MRULE > 0;
   c3  replication: nested minus nested C(M50) > 0 on at least 2 of the 3 other compositions (C2, C3, C4; same val feeds, other member sets), and nested gain over PLAIN > 0 on all 4;
   c4  nested minus fixed MRULE >= 0 under the val reweighted to the private mix (src/pmix.py), and nested over PLAIN >= 0 at the public mix;
   c5  transfer: every gate cell of the rule holds >= 5% of the public-like AND of the private-like test feeds, and the gating variable's test range lies inside the val range; no DAY gate.
 If no family / variant meets c2: verdict = "nothing beats the M-rule family nested by more than its placebo noise"; then the output is the honest nested value of the M-rule family itself:
   C(M50) "holds" iff its nested gain over PLAIN >= +2 paired SE on C1 or p_A <= 0.05, AND > 0 on all 4 compositions; it "is not shown" otherwise.
 At most two candidates are named; if more than two families pass, the two with the largest nested gain over PLAIN that differ in their gate variable are named.
 Deployed weights of a candidate family = its best rule on ALL val feeds of C1 (its honest value = the nested number; the in-sample number is printed next to it).

BLOCK 7. SINGLE DECLARED HYPOTHESES (step 6; NO search; each = one fixed rule read on full val, paired per-feed SE; not nested because nothing is fitted)
 h1  w_ft higher on cold feeds: c = 0.7 where cold share >= 0.45, 0.5 elsewhere. Two definitions of cold share = 2 looks: h1a with CS (pmix variable; 8.4% of val feeds, 29% of test), h1b with CL (the task's definition; 25.9% of val,
     48% of test). Each on a = 0.2 (vs PLAIN) and on the MRULE weights (vs MRULE). SUPPORTED iff delta >= +1 SE on C1 and > 0 on all 4 compositions and >= 0 at pmix-private; REFUTED iff delta <= -1 SE on C1; else NOT SHOWN.
 h2  M-rule with 0.5 on M >= 50 when the popularity-light pool has 8 members. What exists at the time of the read is used: held-out W16 + (4 wide + 4 d128) = W16P20x8 and W8P20x8; full-data FD16 / FD8 + 8 if their files exist when
     the final read is made (FD16P20w8 or FD16P20x8). Read: rule(0.2 / 0.5) minus rule(0.2 / 0.4) and minus PLAIN. SUPPORTED iff (0.5 minus 0.4) >= +1 SE on both 16-base compositions read and >= 0 at pmix-private; REFUTED iff <= -1 SE
     on either; else NOT SHOWN. Prior exposure [D]: the lead's dose line (4-member pools) 0.4 / 0.5 = +0.00043 / +0.00052 (8 + 4) and +0.00071 / +0.00080 (16 + 4).
 h3  M-rule in one branch only: h3-TO = (a_to by the M-rule, a_ft = 0.2), h3-FT = (a_ft by the M-rule, a_to = 0.2). Read each minus PLAIN and minus MRULE (both branches). A single-branch form REPLACES the M-rule iff it is
     >= MRULE + 1 SE on C1 and > MRULE on the 3 others; "the effect sits in branch X" is stated iff branch X alone carries >= 2/3 of the MRULE gain on all 4 compositions.
 h4  three-step M rule on M < 30 / 30-99 / >= 100: h4a = (0.2, 0.3, 0.4) (interpolating form), h4b = (0.1, 0.3, 0.5) (linear ramp); 2 looks. Each minus PLAIN and minus MRULE. SUPPORTED over the M-rule iff >= MRULE + 1 SE on C1 and
     > MRULE on at least 2 of the 3 others; else NOT SHOWN. The fitted three-step forms are families G(M3r) / G(M3q) of the nested search.
 Multiplicity of this block: 2 (h1) x 2 bases + 1 (h2) + 2 (h3) + 2 (h4) = 9 fixed rules.

BLOCK 8. ROBUSTNESS (step 5) for the families named in the output (and for C(M50) in any case): nested gain by val day, by M bucket (pmix bins), under pmix public / private weights; label-free on test: share of
 public-like / private-like test feeds per gate cell against val; share of test feeds whose top-1 item is unchanged against the PLAIN file of the same members, and the mean per-feed Spearman of the two rankings.
 No file under subs/ or preds/ is written; score arrays, if any, are written here as NOT_A_SUBMISSION_*.npy.

=== ADDENDUM 1 ; written 2026-11-15T13:22:28Z AFTER the first nested tables (C1-C4), the hypotheses read and the first placebo draws were seen; everything below is POST HOC in design (not in outcome: no number of it exists yet) ===
 Correction of a count: the family list has 95 eligible families (the DAY gate appears in 5 forms: C, D, E, F, I), not 93. No definition changed.
 Seen so far [M]: nested C(M50) minus PLAIN = +0.00018 / +0.00021 / +0.00055 / +0.00028 on C1..C4; the top C1 family F(Hq) (+0.00076) is negative on C3 / C4; h1-h4 read (logs_hyp.txt).
 X-MODEL read (motivation: the fixed M-rule was chosen on the held-out twins and is read on the full-data members over the SAME val feeds; how much of such a "replication on other trained models" is shared feed noise?):
   for a family, the rule is fitted on member set SRC and read on member set TGT, in two ways:
     xm-same  = fitted on ALL val feeds of SRC, read on all val feeds of TGT (other models, same feeds: the kind of evidence on record for the M-rule);
     xm-nest  = fitted on the training users (4 folds) of SRC, read on the held-out users of TGT (other models AND other users), 5 x 5 folds as before.
   Pairs SRC -> TGT: C3 -> C1, C1 -> C3, C4 -> C2, C2 -> C4. Reported per family: own in-sample, xm-same, xm-nest, own nested (all minus TGT's PLAIN, paired SE for the nested ones).
   POOLED read: rule fitted on the SUM of the training tables of C1 and C3 (resp. C2 and C4), read on the held-out users of each (pool-nest): the selection sees two member sets, the evaluation other users.
   Use: descriptive calibration of "same feeds, other models" evidence, and a second honest read of each family. No candidate can pass on these reads alone; the criteria of block 6 stay as declared.
   Additional observation rule (fixed now): shared-feed share = 1 - mean over families (eligible, 2-3 parameters) of (xm-nest) / mean of (xm-same), reported with both means.

=== ADDENDUM 2 ; written 2026-11-15T13:25:50Z : STOP RULE OF BLOCK 5 TRIGGERED on the W16 placebo -> leak check declared before it is run ===
 Seen [M]: PL-A on FD16 (200 draws): pooled placebo nested mean -0.00002 (sd of the per-draw mean / sqrt(200) = 0.00001). PL-A on W16 (100 draws): pooled placebo nested mean +0.00006 (0.00002): positive beyond "2 SE of the mean".
 Two readings: (i) a leak in my nested procedure; (ii) no leak: the "SE over draws" is not the SE of that mean (all draws share the 14,423 val feeds, so the mean over draws keeps the feed-sampling noise), and / or a real
 constant re-tune effect of the placebo pools (the pseudo pool of 4 same-recipe members deserves the member-equal weight 0.25-0.3, PLAIN gives it 0.2).
 LEAK CHECK (decides between them; criteria fixed now):
   L1 label-permutation null: on the C1 pools, the labels are permuted WITHIN each feed (number of positives per feed kept), 60 independent permutations; the identical search. Under no leak every rule has the same expected NDCG,
      so the nested gain over PLAIN has expectation exactly 0 and the permutations are independent. PASS (no leak) iff |mean over permutations of the pooled nested gain (95 eligible families)| <= 2.5 SE (SE = sd over permutations / sqrt(60))
      AND the same for the variants B-all, B-3, G. The in-sample gains must be positive (they measure pure selection optimism).
   L2 explanation check on the W16 placebo (40 new draws): mean over draws of NDCG(a, c = 0.5) - NDCG(a = 0.2, c = 0.5) for a in the grid, and of NDCG(a = 0.2, c) - PLAIN; reading (ii) is supported iff some a > 0.2 has a positive mean
      of at least the size of the placebo nested mean (+0.00006) with the sign in >= 70% of the draws.
   If L1 FAILS: every nested number of this study is withdrawn until the leak is found. If L1 PASSES: the placebo p-values stand as computed; the wording "placebo mean <= 0" is replaced by "placebo mean = the small real
   constant re-tune effect of the placebo pools plus feed noise", and the p-values are conservative or neutral with respect to it (they compare against the placebo distribution as it is).

=== ADDENDUM 2b ; written 2026-11-15T13:27:22Z ===
 Seen [M]: L1 (60 label permutations): pooled nested gain +0.000036 +- 0.000025 (z +1.44), variants z +1.06..+1.28: PASS by the declared bar (|z| <= 2.5), but the sign is positive in all four statistics (they share noise).
 L2: W16 placebo, const a at c = .5: a 0.1 +0.00007 [80% of draws], a 0.4 +0.00007 [70%]; FD16 placebo: a 0.4 +0.00006 [62%], a 0.1 +0.00003: reading (ii) supported for W16 by the declared rule (a 0.4: +0.00007, 70%).
 To settle L1 with a tighter SE: 240 MORE independent label permutations (seeds 2000..2239); the verdict is taken on all 300: PASS iff |z| <= 2.5 for the pooled mean and for each variant. A bias of the size seen
 (+0.00004) would be one tenth of the effects under study; it is reported in any case.

=== ADDENDUM 3 ; written 2026-11-15T13:27:54Z : TWO-MEMBER-SET statistic with a MATCHED placebo (post hoc design, declared before any of its numbers) ===
 Motivation: each composition alone has a nested paired SE of about 0.0003 and the member sets disagree; the mean over the two 16-base member sets (C1 full-data, C3 held-out twins; same val feeds) halves the member noise.
 T2(family) = mean of the nested gain over PLAIN on C1 and on C3 (variant A, as declared; paired SE from the per-feed mean of the two nested vectors). T2(fixed MRULE) likewise from the fixed rule.
 Matched placebo PL-A2: 100 draws; in each draw the SAME member indices (the i-th FD16 member and the i-th W16 member are twins: same recipe and seed) form the pseudo popularity-light pool (4) / base (12) and the same
 indices swap roles, in both member sets; the identical search on each; T2 of the draw = mean of the two nested gains. p2 = (1 + #{T2_d >= T2}) / (1 + 100). Family-wise: max over the 95 eligible families of T2 per draw.
 Statements fixed now: "the M-gated popularity-light weight is supported across member sets" iff p2(C(M50)) <= 0.05. A family "beats the M-rule family across member sets" iff
 T2(family) - T2(C(M50)) > the 95th percentile of the same difference over the PL-A2 draws; with 94 other families looked at, the family-wise version (max over families of the difference per draw) is printed next to it.
 DOSE TABLE (dose.py; descriptive, fixed rules, in-sample, no selection claimed): final val of constant a in the grid at c = 0.5, by M cell, minus a = 0.2, for C1..C4; pool-only levels by M cell.

=== ADDENDUM 4 ; written 2026-11-15T13:30:23Z : descriptive w_ft curve under the pmix weights (no number of it seen yet) ===
 Why: val OOF under-reads the fine-tuned branch against public (+0.00009 val, about +0.0005 public, +0.00066 / +0.00104 at the public / private mix [D]); the declared h1 reads only c = 0.7 on cold feeds.
 cdose.py prints, for C1..C4, the final val of constant c in {0 (train-only branch alone), 0.3 .. 0.8, 1 (fine-tuned branch alone)} minus c = 0.5, at a = 0.2 and under the M-rule weights, unweighted and under the
 pmix public / private weights, and for the cold cell CS >= 0.45 alone. Fixed rules, in-sample, descriptive. Statement fixed now: "a higher constant w_ft is a candidate for a public probe" iff some c in {0.6, 0.7}
 reads >= +1 SE at the private mix on C1 AND is > 0 at the private mix on all four compositions; otherwise "w_ft = 0.5 stands on val at every mix".

=== ADDENDUM 5 ; written 2026-11-15T13:34:10Z : POST HOC descriptive contrast, full-data members vs held-out twins on short feeds ===
 Seen [M] (dose table + logs_fd_vs_twin.txt, 6 contrasts looked at): a 0.4 minus 0.2 on M < 50 reads +0.00053 (C1) / -0.00001 (C2) on full-data members and -0.00119 / -0.00122 on the twins; paired full-data minus twins
 +0.00172+-0.00067 (z 2.55; 16-base) and +0.00121+-0.00072 (z 1.69; 8-base). Post hoc; the largest of 6 contrasts.
 One decomposition, declared now: which pool carries it? Cross compositions X1 = FD16 base + held-out popularity-light pool (P1w), X2 = W16 base + full-data popularity-light pool (FDP); the contrast "a 0.4 minus 0.2 on M < 50"
 on C1, X1, X2, C3. Reading: "follows the popularity-light pool" iff C1 and X2 are both above X1 and C3; "follows the base pool" iff C1 and X1 are both above X2 and C3; else "not attributable".
 Status of anything found here: a lead for a test on INDEPENDENT full-data students (the 8 new wide ones, s955-962), never a candidate by itself.

=== ADDENDUM 6 ; written 2026-11-15T13:38:06Z : four more compositions (8-member popularity-light pools), after the h2 re-read ===
 The full-data d128 popularity-light fine-tunes (abl_fd10_pop1kd05t25 s951-954) arrived at 13:35Z; hyp.py was re-run as declared in block 7 (h2 on what exists at the final read). Seen [M]: fixed M-rule minus PLAIN with the
 8-member full-data pool (4 wide + 4 d128): FD16 base -0.00010+-0.00020 (pmix-private -0.00035+-0.00034), FD8 base +0.00047+-0.00020; rule 0.2 / 0.5 minus 0.2 / 0.4: FD16 +0.00013+-0.00014, FD8 -0.00019+-0.00014.
 h2 verdict by the declared rule: REFUTED (W16P20x8 at -1.41 SE), unchanged; the FD16P20x8 leg reads +0.9 SE.
 Declared now, before running: the SAME nested search (variant A; no new families, grids or folds) on C5 = FD16 + (4 wide + 4 d128 full-data), C6 = FD8 + the same pool, C7 = W16 + (4 wide + 4 d128 held-out), C8 = W8 + the same.
 Purpose: nested value of the M-rule family C(M50) on 8 compositions instead of 4, and whether any family that failed replication (F(Hq) ..) or the M-rule family changes its picture. No new candidate criteria: a family is
 "confirmed on the 8-member pools" iff its nested gain over PLAIN is > 0 on all of C5-C8; the verdict of the original declaration is not re-opened by these reads (they share feeds and 20 of 24 members with C1-C4).

```
