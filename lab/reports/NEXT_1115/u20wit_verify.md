# u20wit_verify: adversarial review of "fold-0 transductive witness for U20"

Written 2026-11-15 04:15 UTC by the u20wit_verify agent (CPU only; 03:51-04:15 UTC). Reviewed: `reports/NEXT_1115/u20wit.md` and `nn_x/w12/next/u20wit/`.
± = 1 SE paired per feed (ddof 1); n = feeds. [M] measured by me with my own code, [I] inferred. "Member delta" = per-feed mean over seeds 951, 952 of NDCG(B, seed) - NDCG(A, seed); "2-pool" = NDCG of the mean per-feed z of the two seeds.

## Verdict

1. **CONFIRMED.** Every number the verdict rests on reproduces to the last printed digit with my own code (own NDCG, z, fold hash, holdout draw; own CPU inference path built from the flat memmaps on a 4,700-feed sample: max logit difference 1.5e-05 against the study's cache on all six U20 checkpoints) [M]. The fixed rule, applied as written to the cross-user member deltas, reads **2 MORE SEEDS**: B - A holdout (n 24,026) +0.00061 ± 0.00038 selected, +0.00040 ± 0.00038 @last; val (n 13,571) +0.00099 ± 0.00064; ratio 0.43 / 0.30 [M].
2. **The verdict is as fragile as a verdict can be.** @last is +0.0004026, 0.0000026 above the KILL bar; the same rule reads KILL on the 2-pool statistic (-0.00007 ± 0.00044) and on all 30,000 feeds (+0.00027 ± 0.00034) [M]. The study states this. The lead's 03:13:18Z block reads KILL on the pooled column and takes "2 more seeds" by the owner's directive: same branch either way.
3. **Transductive witness: reproduced; "no feed-specific effect established" stands** [M]. I add one measurement that sharpens it: B's extra match to the teacher is the same on feeds it trained on (W0L +0.00638 ± 0.00004), on the same users' unseen feeds (H0 +0.00650 ± 0.00009) and on other users' holdout feeds (XH +0.00649 ± 0.00008, n 8,000) [M]. B is more teacher-like everywhere; nothing is specific to a feed or to a user [I].
4. **Not excluded by the study: "more teacher weight".** kd08w192_s951 minus kd05w192_s951 (teacher share 0.8 against 0.5, one seed, no unlabelled feed) reads val +0.00162 ± 0.00075 and cross-user holdout +0.00069 ± 0.00048 [M]: the size of B - A. Until kd08w192 x8 is read, B's member gain is not evidence for unlabelled inputs [I].
5. **One statement of the study is out of date:** "the rule for the 4-seed read is not written anywhere". The lead wrote it at 03:13:18Z (4-pool B - A >= +2 SE on val AND >= -1 SE on the holdout AND cross-user columns agree in sign with val). The study's "P about 0.66" refers to the old bars, and `run_seeds.sh` prints the old member-rule line at 4 seeds.
6. No label leak, no later snapshot of a user as an input, no GPU process, no read-only file modified by the study [M, within what mtimes and process lists show].

## 1. Claim by claim

### 1.1 Declared read (stored GPU train-only predictions; my code: `v_read.py`, `out_v_read.txt`)

| claim | study's number | my number | agree |
|---|---|---|---|
| holdout cross-user, B - A member, selected (n 24,026) | +0.00061 ± 0.00038 (+0.00001, +0.00121) | +0.0006107 ± 0.0003838 (+0.00001, +0.00121) | yes |
| holdout cross-user, B - A member, @last | +0.00040 ± 0.00038 (exactly +0.0004026) | +0.0004026 ± 0.0003825 (-0.00040, +0.00121) | yes |
| val cross-user, B - A member, selected (n 13,571) | +0.00099 ± 0.00064 (+0.00181, +0.00017) | +0.0009887 ± 0.0006366 (+0.00181, +0.00017) | yes |
| val cross-user, B - A member, @last | +0.00085 ± 0.00064 | +0.0008471 ± 0.0006371 | yes |
| holdout cross-user, C - A member, selected / @last | +0.00143 ± 0.00040 / +0.00133 ± 0.00040 | +0.0014260 ± 0.0004034 / +0.0013290 ± 0.0004005 | yes |
| ratio (B - A) / (C - A), holdout cross-user, selected / @last | 0.43 / 0.30 | 0.428 / 0.303 | yes |
| holdout cross-user, 2-pool B - A, selected / @last | -0.00007 ± 0.00044 / -0.00036 ± 0.00044 | -0.0000713 ± 0.0004386 / -0.0003630 ± 0.0004410 | yes |
| holdout cross-user, 2-pool C - A, selected | +0.00073 ± 0.00045 | +0.0007277 ± 0.0004456 | yes |
| val cross-user, 2-pool B - A, selected | +0.00120 ± 0.00074 | +0.0012037 ± 0.0007413 | yes |
| holdout fold-0 users, B - A member (n 5,974) | -0.00111 ± 0.00075 | -0.0011136 ± 0.0007539 | yes |
| val fold-0 users, B - A member / 2-pool (n 852) | +0.00483 ± 0.00237 / +0.00763 ± 0.00265 | +0.0048308 ± 0.0023751 / +0.0076283 ± 0.0026471 | yes |
| holdout all, B - A member / 2-pool (n 30,000) | +0.00027 ± 0.00034 / -0.00017 ± 0.00039 | +0.0002673 ± 0.0003421 / -0.0001708 ± 0.0003880 | yes |
| val all, B - A member / 2-pool (n 14,423) | +0.00122 ± 0.00062 / +0.00158 ± 0.00071 | +0.0012157 ± 0.0006152 / +0.0015832 ± 0.0007149 | yes |
| holdout cross-user without the user-0 bucket (n 22,796), selected / @last | +0.00074 ± 0.00039 / +0.00054 ± 0.00039 | +0.0007373 ± 0.0003919 / +0.0005401 ± 0.0003902 | yes |
| val, strict hash split (n 11,727), selected / @last | +0.00106 ± 0.00068 / +0.00087 ± 0.00068 | +0.0010629 ± 0.0006780 / +0.0008668 ± 0.0006820 | yes |
| counts: holdout fold 0 / cross-user; val fold-0 users / cross-user | 5,974 / 24,026; 852 / 13,571 | 5,974 / 24,026; 852 / 13,571 (val feeds with a train user 4,386 + 658 user-0 feeds) | yes |
| rule on the cross-user member deltas | 2 MORE SEEDS (selected and @last) | 2 MORE SEEDS (selected and @last) | yes |
| A/A null, 2 seeds per arm, selected: sd; share >= +0.00061 | 0.00029 / 0.00025; 0.7% / 0.0% | 0.00029 / 0.00025; 0.71% / 0.00% (3 and 0 of 420 arrangements of 8 members) | yes, see issue 5 |
| 2-pool gain of the A pair on the cross-user holdout against B, C | +0.00159 against +0.00090, +0.00089 | +0.00159, +0.00090, +0.00089 | yes |

Tie convention [M, `out_v_ties.txt`]: 0.05-0.13% of positives are tied with another candidate; B - A @last is +0.0004026 (pessimistic, the project metric), +0.0004099 (optimistic), +0.0004094 (file order). No convention puts it below the bar; all three are inside the numerical noise of the bar (CPU minus GPU of a delta: +0.00007 ± 0.00019 on XH).

### 1.2 Transductive witness (CPU fp32 against CPU fp32; `v_wit.py`, `v_cmp.py`, `out_v_cmp.txt`)

How I checked the cache before using it [M]: (i) my witness feed lists, built from the two posteriors, the holdout draw and the fold hash, are identical to `cache/sets.npz` (W0S 11,022; W0L 26,883; H0 5,974); (ii) `rows.npy` and the cached labels equal the train flat rows / `cand_target` of those feeds; (iii) I re-scored a random sample (W0S 2,000, W0L 1,500, H0 600, XH 600; 469,307 rows) with a batch builder of my own that reads the flat memmaps directly (no `SplitData`, no sub-split cache): max |logit difference| 1.5e-05 for each of the six U20 checkpoints and xf0_s951, and for the four `rl` files of the short witness. The full-set numbers below are my NDCG on the study's cached logits.

| claim | study's number | my number | agree |
|---|---|---|---|
| W0S short (n 11,022), B - A member (per seed) | +0.00060 ± 0.00082 (+0.00126, -0.00006) | +0.00060 ± 0.00082 (+0.00126, -0.00006) | yes |
| W0S, 2-pool / @last member | -0.00038 ± 0.00097 / +0.00037 ± 0.00083 | -0.00038 ± 0.00097 / +0.00037 ± 0.00083 | yes |
| W0S with the val / test domain token | +0.00070 ± 0.00084 | +0.00070 ± 0.00084 (+0.00142, -0.00002) | yes |
| W0S by M 5-9 / 10-19 / 20-34 / 35-49 | -0.00339 ± 0.00264 / -0.00130 ± 0.00217 / +0.00298 ± 0.00141 / +0.00041 ± 0.00121 | identical | yes |
| W0L long (n 26,883), B - A member (per seed) | +0.00095 ± 0.00037 (+0.00131, +0.00059) | +0.00095 ± 0.00037 (+0.00131, +0.00059) | yes |
| W0L, 2-pool / @last member | +0.00026 ± 0.00042 / +0.00086 ± 0.00037 | +0.00026 ± 0.00042 / +0.00086 ± 0.00037 | yes |
| W0S + W0L (n 37,905), member / 2-pool | +0.00085 ± 0.00035 / +0.00008 ± 0.00041 | +0.00085 ± 0.00035 / +0.00008 ± 0.00041 | yes |
| H0 (n 5,974), member / 2-pool / @last | -0.00086 ± 0.00076 / -0.00100 ± 0.00085 / -0.00036 ± 0.00076 | identical | yes |
| XH (n 8,000), B - A member / 2-pool; C - A | +0.00082 ± 0.00067 / +0.00018 ± 0.00075; +0.00100 ± 0.00069 | identical | yes |
| C - A in-sample bound, W0S / W0L | +0.02286 ± 0.00089 / +0.01780 ± 0.00042 | identical | yes |
| W0L - XH; W0S - XH | +0.00013 ± 0.00076; -0.00022 ± 0.00106 | identical | yes |
| W0L - H0, selected / @last | +0.00182 ± 0.00084 / +0.00122 ± 0.00084 | identical | yes |
| H0 - XH | -0.00168 ± 0.00101 | identical | yes |
| within-feed corr to teacher, B minus A: W0S / W0L / H0 | +0.00769 ± 0.00014 / +0.00638 ± 0.00004 / +0.00650 ± 0.00009 | identical; seen minus unseen (W0L - H0) -0.00012 ± 0.00009; XH +0.00649 ± 0.00008 (new) | yes |
| membership | 0 of u20a's feeds; all unlabelled in u20b, none in its pool; 17,462 of W0L in C's pool | 624,752 fold-0 feeds outside the holdout; pools 107,273 / 107,273 / 134,118; 17,462; 0 witness users (user-0 bucket aside) own a feed in A's training set; logs print the same counts | yes |
| feeds without a positive (the only label-dependent factor of an unlabelled target) | 0 | 0 (W0S, W0L, H0) | yes |

### 1.3 Validation gates

| claim | study's number | my number | agree |
|---|---|---|---|
| teacher file = sum of 3 members; CPU 3-mean against teacher / 3, mean within-feed corr | 0.999936-0.999946 on 5 sets; min 0.983 | 0.999936-0.999946; share >= 0.999: 0.9963-0.9987; min 0.9835; `teacher_xf0.n` = 3; the three xf0 checkpoints carry `xfold 0`, `holdout 30000` | yes |
| base - xf0, member, 3 seeds | W0S +0.02369 ± 0.00090, W0L +0.01707 ± 0.00042, W1S +0.00253 ± 0.00086, W1L +0.00120 ± 0.00087, H0 +0.00152 ± 0.00081 | identical | yes |
| positives per feed, W0S against val M < 50 | 1.066 / 1.123 / 1.237 / 1.322 against 1.078 / 1.157 / 1.250 / 1.359 | identical on both sides (val n 1,043 / 1,505 / 2,030 / 1,315) | yes |
| CPU against stored GPU file on H0, per U20 member | -0.00022 .. +0.00053 | -0.00022 ± 0.00023, +0.00026 ± 0.00022, +0.00002 ± 0.00020, +0.00053 ± 0.00021, -0.00019 ± 0.00022, -0.00013 ± 0.00019 | yes |
| B - A on XH from the GPU files against the CPU replay | +0.00075 ± 0.00066 against +0.00082 ± 0.00067 | +0.00075 ± 0.00066 against +0.00082 ± 0.00067 | yes |
| validation finished before any U20 read | 11-14 23:26Z, first U20 json 00:45Z | file times: `sets.npz` 22:24Z, `wit_validate.txt` 23:10Z, first json 00:45Z | yes |

## 2. Issues, by severity

**1. (medium) The verdict depends on which statistic is called "the" read, and sits on the bar.** The 20:21Z text says "member deltas @last and selected, paired per feed, pooled over the 2 seeds"; the study read it that way and I agree with that reading. Selected: +0.00061 (1.6 SE above zero, 0.5 SE above the KILL bar). @last: +0.0004026, on the bar. The whole selected-minus-@last gap (+0.00021) comes from seed 951 alone: `select_on` is `holdout`, seed 952 selected its last step in all three arms, seed 951 did not [M, `out_v_extra.txt`]. The selected column is therefore chosen on the witness it is read on; @last is the selection-free read. On the 2-pool the holdout shows nothing (-0.00007 ± 0.00044). So "2 more seeds" is correct by the letter and carries almost no information about the sign on the holdout.

**2. (medium) A's pair gains abnormally much from pooling, so the member delta overstates what a pool would see** [M, `out_v_pairgain.txt`]. 2-pool gain on the cross-user holdout: u20a +0.00159, u20b +0.00090, u20c +0.00089. Same-recipe pairs (28 pairs each): kd05w192 mean +0.00110, sd 0.00016, max +0.00145; kd05t25w192 mean +0.00089, sd 0.00021, max +0.00141. A's pair is above all 56 reference pairs; B and C are ordinary. On val cross-user all three are ordinary (+0.00129 / +0.00150 / +0.00125 against reference pairs with mean +0.00126, sd 0.00044 and mean +0.00103, sd 0.00030). The study reports the three gains but did not place them in a reference distribution. Reading [I]: on the holdout a good part of A's member deficit is member noise that pooling removes (A has 20% fewer users and 20% fewer steps), or the A pair is a 1-in-50 draw; two seeds cannot tell. Either way the final files are pools, and the pool column is the one to weigh.

**3. (medium) Alternative explanation not excluded: total teacher weight.** B has teacher share 1.0 on 20% of its feeds (0.6 on average against 0.5). Its extra match to the teacher is +0.0064..+0.0065 on seen feeds, on the same users' unseen feeds and on other users' feeds alike [M]. The one finished control seed, kd08w192_s951 minus kd05w192_s951, reads val (n 14,423) +0.00162 ± 0.00075, holdout (n 30,000) +0.00052 ± 0.00043, cross-user holdout (n 24,026) +0.00069 ± 0.00048; @last +0.00152 ± 0.00076 / +0.00091 ± 0.00044 / +0.00090 ± 0.00049 [M, `out_v_rl_kd08.txt`]. B - A is the same size (val all +0.00122 ± 0.00062, cross-user holdout +0.00061 ± 0.00038). The lead named this control at 03:13Z. The study measured that B is more teacher-like on every feed set and names the step budget (78,710 against 98,350 steps); it does not name total teacher weight as a confound, and the seed-951 comparison was available to it. One seed: suggestive, not a measurement of the control.

**4. (medium) Stale statement and a tool that prints a superseded rule.** The report (03:50Z) says the 4-seed rule is not written anywhere and asks the lead to write it. The lead wrote it at 03:13:18Z in `E_kd_predeclared.txt`: GO iff 4-pool B - A >= +2 SE on val AND >= -1 SE on the holdout AND the cross-user columns of u20wit agree in sign with val; CLOSED iff 4-pool B - A < +1 SE on val. Consequences: (a) the study's "stays between the bars with P about 0.66" is about the old +0.0004 / +0.0008 bars and no longer describes the decision; (b) `u20_read.py` at 4 seeds will print a GO / KILL / 2-MORE-SEEDS line from the old member rule: that line is not the decision; the pool columns of the same output are. (c) Seeds 953 / 954 were already queued by the lead (stopped while waiting at 03:35Z and re-queued behind the staleness arms).

**5. (low) The A/A statement is more exact than its data.** "P(null >= +0.00061) 0.7% / 0.0%" is 3 and 0 of 420 arrangements of the same 8 members (7 degrees of freedom; the statistic's sd equals the member sd). At @last the same null has sd 0.00030 / 0.00038 and exceeds +0.00061 in 1.4% / 4.5%; the @last statistic itself (+0.00040) is exceeded by 9.5% / 15.7% of null arrangements [M]. By the paired SE the selected delta is 1.6 SE and the @last delta 1.05 SE above zero. "Unlikely to be seed noise" holds for the selected column at about 2 sd and does not hold for @last.

**6. (low) The val column of the new 4-seed rule should be the cross-user one.** On val, 28% of the all-feed 2-pool B - A comes from the 852 feeds of fold-0 train users (+0.00763 ± 0.00265 on 5.9% of feeds): val all +0.00158 ± 0.00071 (2.2 SE), val cross-user +0.00120 ± 0.00074 (1.6 SE) [M]. The 21:26Z addendum calls that column the same-user channel and "not evidence for a transfer-set programme". The 03:13Z rule says "val"; if it is read on all val feeds, the same-user channel can carry it over the +2 SE bar. I am not changing the rule; I am pointing at the column.

**7. (low) XH was added after the first witness numbers were seen.** The interim witness read is from 02:45Z, `xh_build.py` from 03:01Z, after W0L - H0 had read +0.0018. XH is a seeded random subsample and the natural baseline, so the risk is small; the conclusion does not depend on it: W0L on the CPU (+0.00095 ± 0.00037) minus the full cross-user holdout from the GPU files (+0.00061 ± 0.00038, n 24,026) is +0.00034 ± 0.00053 (mixes CPU and GPU files, worth about ±0.0002) [M].

**8. (low) Which contrast isolates the feed.** W0L - XH mixes two things (the feed's inputs, and whether A knows the user at all). W0L - H0 holds the users fixed and is the only contrast that isolates the feed: +0.00182 ± 0.00084 selected (2.2 SE), +0.00122 ± 0.00084 @last (1.5 SE). The study reports both and concludes "not established"; I agree, because the label-free teacher-match check contradicts a feed-specific effect at a much tighter SE (seen minus unseen -0.00012 ± 0.00009 against a gain of +0.0064) and because C - A, where C saw these users with labels, is also low on H0 (+0.00038 ± 0.00082 against +0.00100 ± 0.00069 on XH). For the 4-seed read W0L - H0 should be named in advance as the transductive statistic.

**9. (info) Fold-1 control is not "about zero".** base - xf0 on W1S is +0.00253 ± 0.00086 (2.9 SE). The study explains it by base's 25% more users and shows the same +0.0025 on the cross-user holdout. Accepted: the in-sample margin on the fold-0 witness is 9-14 times larger.

**10. (info) Rule 1, as the study itself reports:** light read-outs ran next to its one heavy scoring process, and a few text-editing one-liners ran without `CUDA_VISIBLE_DEVICES=""`. I cannot check this beyond its logs. I found no GPU process of it, no process of it alive, and no file of it outside its directory and its report.

Not found: a label leak (B's target on a fold-0 feed is the teacher softmax times "has a positive"; aux weight 0; no position input with `--no_pos`; the teacher's three members carry `xfold 0`); a use of later snapshots as inputs; an unpaired SE; a moved threshold; a difference between the arms other than `xfold` / `xfold_unlab` / `seed`. Behavioural confirmation that B never fitted fold-0 labels: C's in-sample margin on the witness is +0.018 / +0.023, B's is +0.0006 / +0.0010 [M].

## 3. What the lead should conclude

- By the rule as written: 2 MORE SEEDS. Confirmed. It is the branch already taken at 03:13Z.
- What the two seeds actually say [M]: at pool level B - A is zero on the holdout (-0.00007 ± 0.00044) and +1.6 SE on cross-user val (+0.00120 ± 0.00074). At member level B is ahead by about +0.0006..+0.0010 on every feed set, seen or unseen, own users or others, and one seed of the "more teacher weight" control shows the same size.
- The transductive programme has no measured mechanism: teacher-only training on a feed does not make B match the teacher better on that feed than on feeds it never saw (-0.00012 ± 0.00009). This is the forecast of SYNTHESIS bet 5 (a) (0.9933 / 0.9933), now measured inside U20 itself [M; the conclusion is I].
- For the 4-seed read: decide on the pool columns by the 03:13Z rule, read val on the cross-user column next to the all-feed one, ignore the member-rule line the tool prints, and read kd08w192 x8 first, as the 03:13Z block already says.
- The study's tool is sound: the witness sets, the CPU path, the row maps and the labels all check out. It can be reused for seeds 953 / 954 as it is.

## 4. Method

- Declared read (`v_read.py`): train `feeds.parquet`, the old posterior and `default_rng(12345)` rebuild the 30,000 holdout feeds (M equal to `holdout_meta.npy`; labels equal to `holdout_y.npy` on 3,000 sampled feeds); fold = uid == 0 ? feed index % 5 : (uid * 2654435761 mod 2^32) mod 5; NDCG is my own rank-count formulation with pessimistic ties, z my own loop. Nothing imported from the study, `pc_common`, `src/metrics` or `src/stack_val`. My member NDCG on val equals the `val_ndcg` of the six json logs to 5 digits.
- Witness sample (`v_wit.py`): sets rebuilt from scratch; batches assembled with numpy straight from `data/flat/train/*.npy`, the item-embedding memmap and `nn_x/w5/feat_cache/train/` (same tensor layout as `SplitData.gather`, written independently); the model class is the project's `nn_x/w5/model.py`; fp32, eval mode, `use_pos=False`; domain token [1 - p, p, 0] with the old posterior (0 for M < 50), or no token for the `rl` check.
- Full-set witness numbers (`v_cmp.py`, `v_anchor.py`, `v_rl_kd08.py`): my NDCG / z / within-feed correlation on the study's cached logits, used only after the sample agreed to 1.5e-05; labels read from train `cand_target` at the rows, for evaluation only.
- Extra measurements: `v_extra.py` (run args, selection metric, teacher match on XH), `v_pairgain.py` (2-pool gains against 56 same-recipe pairs), `v_ties.py` (tie convention).

## 5. Limits and what I did NOT check

- I re-scored a sample (4,700 of 69,107 feeds), not the full sets; the full-set witness numbers rest on the study's cache plus that sample agreement. base, xf1 and kd05w192 checkpoints were not re-scored by me (xf0_s951 was).
- Not re-derived: the xf1 - xf0 fold swap, the rule tests on known arms, the bootstrap intervals of the ratio, the per-fold (1..4) holdout split, the splits of W0L by cooldown-pool membership and by M, the short witness re-weighted to the val size mix, the A/A null of a 2-pool delta and the 4-seed null, the 140 low-correlation feeds one by one.
- The kd08w192 control is ONE seed (951); the 8-seed read was not available.
- A's abnormal 2-pool gain is one pair; I cannot say whether it is structural.
- I did not run the lead's `arm_read8.py` on the U20 arms and did not compare with the lead's 03:13Z block beyond the pooled numbers it prints (val +0.00158 ± 0.00071, holdout -0.00017 ± 0.00039: equal to mine).
- I did not verify the study's self-reported rule-1 deviations or its stopped PIDs; I only see that nothing of it runs now.
- File modification times show nothing changed by the study outside its directory; I cannot attribute every file changed by other agents in the same hours (reports/STALE_1115, reports/CATBOOST_1115, nn_x/w12/stale, nn_x/w12/cb, nn_x/w12/kd read scripts).
- No GPU replay; nothing about test or public.

## 6. Audit of files and processes (look only)

- `nn_x/w5/train.py` 2026-11-14 20:20:00Z (expected). `nn_x/w5/lane.sh` 2026-11-15 03:35:49Z: later than the expected 22:14Z, and it matches the lead's own block "03:35:48Z ... QUEUE REORDERED" in `E_kd_predeclared.txt`; not the study. `data.py`, `model.py`, `finetune.py`, `common.py`, `cd_arms*.py`: 11-12 / 11-13. Nothing under `src/` or `nn/` after 21:50Z. `nn_x/w5/__pycache__` unchanged since 20:20Z.
- `nvidia-smi --query-compute-apps`: 32 processes, all `train.py` / `finetune.py` of the lanes (kd08w192, kwd2, kwsb2, kwwd10, hz and ref re-cools, one pop1 fine-tune); none from `next/u20wit`. `pgrep -af next/u20wit`: nothing.
- My own compliance: CPU only; one heavy process at a time (two 12-thread scoring passes of 6 and 4 minutes, PIDs 4118307 and 4130351, both ended by themselves); read-outs with 4-6 threads; no process signalled; nothing written outside `nn_x/w12/next/u20wit_verify/` and this report. Literal deviations from rule 1, stated plainly: one inline python that printed a json and two npz headers (numpy only), one `python3` one-liner that edited my own `v_wit.py`, and the `nvidia-smi` look ran without `CUDA_VISIBLE_DEVICES=""`; none imported torch.

## 7. Commands

```
cd <repo>
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 NUMBA_NUM_THREADS=12 POLARS_MAX_THREADS=12
PY=.venv/bin/python; D=nn_x/w12/next/u20wit_verify
$PY $D/v_read.py > $D/out_v_read.txt                 # declared read, own code (1 min)
$PY $D/v_wit.py build > $D/out_v_wit_build.txt       # witness sets against the study's, sample draw
setsid nohup $PY $D/v_wit.py score w5_e10_u20a_s951 w5_e10_u20b_s951 w5_e10_u20a_s952 w5_e10_u20b_s952 w5_e10_u20c_s951 w5_e10_u20c_s952 w5_e10_xf0_s951 > $D/out_v_wit_score.log 2>&1 < /dev/null &   # 6 min
DOM=rl setsid nohup $PY $D/v_wit.py score w5_e10_u20a_s951 w5_e10_u20b_s951 w5_e10_u20a_s952 w5_e10_u20b_s952 > $D/out_v_wit_score_rl.log 2>&1 < /dev/null &   # 4 min, after the first
$PY $D/v_cmp.py > $D/out_v_cmp.txt                   # sample against cache, transductive read, teacher, known answers, A/A null
$PY $D/v_extra.py > $D/out_v_extra.txt               # run args, selection metric, teacher match on XH
$PY $D/v_pairgain.py > $D/out_v_pairgain.txt         # 2-pool gains against same-recipe pairs
$PY $D/v_anchor.py > $D/out_v_anchor.txt             # CPU against stored GPU files
$PY $D/v_rl_kd08.py > $D/out_v_rl_kd08.txt           # rl token on the short witness; kd08w192_s951 control
$PY $D/v_ties.py > $D/out_v_ties.txt                 # tie convention
```

## 8. Files I created

All under `nn_x/w12/next/u20wit_verify/` (plus this report):
- scripts: `v_read.py`, `v_wit.py`, `v_cmp.py`, `v_extra.py`, `v_pairgain.py`, `v_anchor.py`, `v_rl_kd08.py`, `v_ties.py`
- outputs: `out_v_read.txt`, `out_v_wit_build.txt`, `out_v_wit_score.log`, `out_v_wit_score_rl.log`, `out_v_cmp.txt`, `out_v_extra.txt`, `out_v_pairgain.txt`, `out_v_anchor.txt`, `out_v_rl_kd08.txt`, `out_v_valpos.txt`, `out_v_ties.txt` (and two empty `.err` files)
- caches: `cache/sample.npz`, `cache/my_<variant>.npy` (7 files), `cache/my_<variant>__rl.npy` (4 files)
- `PROGRESS.md`
- report: `reports/NEXT_1115/u20wit_verify.md`
