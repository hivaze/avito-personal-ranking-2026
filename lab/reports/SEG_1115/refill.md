# SEG_1115 / refill: the refill replay (zero training) for histories at the 256-event cap

Written 2026-11-15T18:21Z. Directory `nn_x/w12/seg/refill/`. Rule: `nn_x/w5/logs/E_kd_predeclared.txt`, block 2026-11-15T17:41:25Z (unchanged). Declaration: `DECLARATION.txt` block 0 at 17:57:37Z and block 1 at 18:01Z, both before the first labelled number (18:11Z); blocks 2-3 record what was added after the read. Marks: [M] measured here, [D] arithmetic from [M], [I] judgement. SE are cluster-robust by user. Sets: the 30,000 holdout feeds (decision) and val (sign). Members: 2 distilled wide students + 2 hard-label members, all held-out.

## 0. Verdict

**UNDECIDED** by the declared rule.

1. **The verdict row** [M] (the rule's passes with the events OLDER than the stored 256th event; 4-pool; holdout, 2,358 feeds): numerator N(c) - N(b) = +0.00117 +- 0.00082 (z 1.43), denominator N(a) - N(b) = +0.00262 +- 0.00154, **rho = 0.447** [bootstrap 95% -1.40, +2.80]; val numerator +0.00269 +- 0.00157 (same sign). Not OPEN (rho below 0.5, z below 2). Not CLOSE (upper bound of the numerator +0.00277 against a quarter of the denominator, +0.00065).
2. **A second design was declared before any labelled number and run in the same job** (all events the earlier snapshots hold and the stored history lacks; 3,123 holdout feeds): numerator +0.00151 +- 0.00093 (z 1.62), rho 0.632 [-1.21, +3.40]; val +0.00267 +- 0.00215. UNDECIDED as well. The rule's last clause reads "one more replay design at most, then closed": if design 2 counts as that design (the lead's call), the door closes without an arm.
3. **Why the rule could not decide** [M / D]: 3,235 holdout feeds qualify (10.8% of the holdout; the review's power estimate assumed several times more). On them the denominator itself is 1.7-1.8 SE, so OPEN needed rho >= 0.63-0.78 and CLOSE needed old events that hurt. No larger sample exists: holdout and val together (every labelled feed on which these members are out of sample) give +0.00173 +- 0.00078 (z 2.23, rho 0.66 [-0.13, +2.94]) for design 1 and +0.00194 +- 0.00099 (z 1.96, rho 0.65 [+0.01, +1.81]) for design 2.
4. **What the replay does show** [M]: (i) recovered events are worth what the stored events in the same slots are, within +-0.001 (recovered minus stored -0.0004 +- 0.0010 and -0.0001 +- 0.0010, pooled); (ii) it is the user's own past: another user's events in the same slots read -0.0011 +- 0.0007 against empty slots and the user's own beat them by +0.0028 +- 0.0009 (z 3.0) and +0.0031 +- 0.0011 (z 2.7); (iii) the numerator is positive in 16 of 16 member x set x design cells; distilled and hard-label families do not differ (+0.0016..+0.0019 on the holdout, z 1.8-2.1).
5. **Models not trained for it get nothing reliable from a 384-event history** [M]: stored 256 + recovered events minus stored 256 = +0.0009 +- 0.0006 (design 1) and 0.0000 +- 0.0008 (design 2), pooled.
6. **Size at pool level** [D]: +0.00015 +- 0.00007 (design 1) to +0.00035 +- 0.00015 (design 2) on private-like test days IF a retrained model captured the whole worth measured in trained slots; +0.0001 +- 0.0001 without retraining. The 14.3% on record is verified (14.35% of test, private-like days 14.05%), but only 8.2% (design 1) to 10.9% (design 2) of private-like feeds actually receive events.
7. **Three facts that differ from the brief** [M]: (a) the final members DO carry a positional code of history events: `rec = nn.Embedding(256, d)` is added to every history token and is trained (`--no_pos` removes the display-position tower only), so a 384-event history needs a model change and retraining; (b) a history is not a shifted log (per-type caps): events older than the stored 256th are few (median 29; 128 of them exist for 6-7% of qualifying feeds), and on val / test about two thirds of the events earlier snapshots add are NOT older than the 256th event (mostly views the per-type cap removed); (c) the alignment misplaces 0.4-0.5% of events, yet only 60-72% of recovered events are events a deeper cap would have stored: a refilled history is not the history a deeper cap would have given.
8. **Recommendation** [I]: by the rule, no arm. The mechanism is more likely real than not, the prize is small (probability of a pool gain >= +0.0003 from a 16-student arm about 0.15) and train rows cannot be built like test rows (median gap to the earlier snapshot 2.9 days on train-period feeds against 14.2 on test). If idle GPU-hours are spent at all: a 4-snapshot re-cool pilot (about 1 GPU-hour) under the bucket kill rule of section 6, nothing larger.

## 1. What was run, and three facts that differ from the brief

Directory `nn_x/w12/seg/refill/` (scripts, outputs `out_*.txt`, `DECLARATION.txt` with its blocks stamped before the first labelled number, `PROGRESS.md`). The rule is the block of 2026-11-15T17:41:25Z in `nn_x/w5/logs/E_kd_predeclared.txt`, unchanged.

**Members (held-out; `args.holdout = 30000` in each checkpoint, so none trained on the holdout feeds):** distilled wide students `w5_e10_kd05w192_s951`, `w5_e10_kd05t25w192_s951` (d192); hard-label members `w5_e10_base_s951`, `w5_e10_base_s952` (d128). Pool = member-equal mean of per-feed score z (`nn_x/w5/blend.py`; the project pools score z, not rank z). Inference only, fp32, one process on GPU N under the lane lock (launched 18:01:24Z with 3 compute processes and 88.7 GB free on the card; 9.8 minutes; 20 passes x 4 members x 2 sets). Own subset loader (`s2_replay.py`) that mirrors `SplitData.gather`; pass (a) reproduces the stored predictions of every member on both sets (flat correlation 0.999997-1.000000; the stored ones are bf16) [M].

**Facts found before any labelled number (all in DECLARATION block 0):**

1. **The members do use a positional code for history events.** `--no_pos` switches off the display-position tower only. Every history token gets `rec = nn.Embedding(256, d)` added (trained, row norm 2.4-3.0 in all four members), and the retrieval values carry a log2 bucket of the event's position. Other order / length inputs: max cosine over the newest 8 and 32 events, recency of the newest same-category event, log(1 + H), category count / H, user_id-0 flag x log(1 + H) (zero here). Consequences: pass (c) shows old events with the position code of slots 129-256, as the rule intends; pass (d) cannot run as is (position 256 does not exist) and needed a declared convention.
2. **A history is not a shifted log** (per-type caps, section 2), so "the events beyond the cap" are two different things: events older than the stored 256th (T, few) and events the per-type caps removed (the rest of U, many).
3. **The decision sample is small**: 3,235 qualifying holdout feeds.

**Designs (both declared before any labelled number).** Design 1 = the rule's passes with T (literal). Design 2 = the same passes with U, declared because design 1 cannot fill the slots (median 29 events) and because U is what an extended-history arm would actually feed.

**Passes** (the model sees the listed events as its history; every in-model history input follows them, exactly as the loader's `max_hist` does):

| pass | history shown |
|---|---|
| a | the stored 256 events |
| b | the newest 128 (H = 128) |
| c | the newest 128 + the first n = min(recovered, 128) recovered events in slots 129..128+n (H = 128 + n) |
| am | the stored newest 128 + n events (length-matched stored reference; = a when n = 128) |
| e (placebo) | the newest 128 + n recovered events of another user's read-set feed (same set, H = 256, at least n recovered events, fixed seed) |
| b192 / c192 | the newest 192 / the newest 192 + min(recovered, 64) recovered events (the review's second line) |
| g (calibration) | the newest 128 + the stored events 193-256 shown in slots 129-192 (H = 192), read against b192 |
| d1 | stored 256 + n recovered events as 256 + n tokens; `rec` index held at 255 beyond position 255; every in-model dense input held at its pass-(a) value |
| d2 | as d1, dense inputs recomputed over all events with H held at 256 |
| dP | d1 with the other user's events |

**Inputs that depend on the history and were NOT recomputed in any pass** (so every effect is a lower bound in magnitude, as in the review's truncation replay): the precomputed xt columns `seq_cv_invd20 / pop / norm` (co-visitation with the feed's history), `ctx_nd_*` (10, in-feed context against the history and the user embedding), `ctx_hn_*` (6, competition for the last 32 history items), `ctx_hs_*` (4), `pre2_fe_z / new_z / fe_s / new_s / hmax32` (trajectory-encoder scores over the history): up to 28 of the 34 columns were computed on the stored 256 events and stay as stored. The 5 `it_hist_*_luo` popularity columns do not depend on the feed's own history. The organisers' user embedding is unchanged too.

**Statistics.** Per-feed NDCG by `src/metrics.py`; differences paired per feed; SE cluster-robust by user; ratios with a 2,000-draw user bootstrap. rho = [N(c) - N(b)] / [N(a) - N(b)] (the rule). Because most feeds cannot fill 128 slots I also declared: ceiling = [N(am) - N(b)] / [N(a) - N(b)] (the rho that perfect events of the same count would give) and the length-matched share rho_m = [N(c) - N(b)] / [N(am) - N(b)]. Verdict line = design 1, literal rho on the holdout, sign on val.

## 2. Alignment (step 1; label-free; `align_lib.py`, `s1_align.py`, `s1b_poscheck.py`, `out_s1.txt`, `out_s1b.txt`)

**Stored order.** Index 0 = newest. Three sources agree: the organisers' readme ("ordered from new events to old"), the docstring of `data.py`, and the data [M]: against the most recent earlier snapshot, the events only the later history holds sit before the first common event (36.9 per feed, against 12.4 after the last common one), and the events only the earlier snapshot holds sit after its last common event (25.9, against 0.4 before the first). So events beyond the cap continue the stored array at its END, oldest last.

**A history is not a shifted log** [M]. The exact-shift test (later[k : k + n] == earlier[0 : n]) succeeds on 5.2% of pairs. Two reasons: (i) per-type caps (recorded as INTEGRITY-1 in `reports/review_integrity.md`: a snapshot keeps the newest N events of each type, about 100-110 views in val / test and up to about 210 in half of the capped train feeds, then the newest 256 of the union), so a view can leave the history while older events of other types stay; (ii) simultaneous events (same item, several types) change order between snapshots. I therefore align by a longest common subsequence of (item, type, source), repair tie swaps, and merge all strictly earlier snapshots (most recent first) into one union timeline.

Two sets of recovered events follow, and the difference matters for everything below:

- **T** = events of the union timeline after the stored 256th event: the literal "events immediately older than the stored 256th".
- **U** = every event the earlier snapshots hold and the stored history lacks (old-only), newest first. T is its tail; the rest are events NEWER than the stored 256th that the per-type caps removed (mostly views).

**Criteria.** A (aligned): at least one earlier snapshot shares >= 16 LCS anchors and an exact contiguous run >= 8 with the timeline; feeds failing A are dropped. B (end-anchored, needed for T only): the deepest stored event that is an anchor has index >= 248.

| [M] | holdout (30,000) | val (14,423) | test (28,595) |
|---|---|---|---|
| capped feeds (H = 256) | 11,601 (38.67%) | 4,176 (28.95%) | 8,239 (28.81%) |
| qualifying: capped, user_id != 0, >= 1 strictly earlier snapshot | **3,235 (10.78%)** | **2,160 (14.98%)** | **4,102 (14.35%)**; public-like days 15.03%, private-like 14.05% |
| earlier snapshots per feed (median / 95%); where they come from | 1 / 3; train only | 1 / 4; train 4,082, val 13 | 1 / 4; train 7,533, val 55, test 53 |
| gap to the most recent earlier snapshot, median days | 2.9 | 9.6 | 14.2 |
| criterion A pass / FAIL (dropped) | 3,140 (97.1%) / 2.9% | 1,879 (87.0%) / 13.0% | 3,222 (78.5%) / 21.5% |
| criterion B pass (of qualifying) | 2,511 (77.6%) | 1,465 (67.8%) | 2,565 (62.5%) |
| T events per feed (A and B), 5 / 25 / 50 / 75 / 95% | 0 / 8 / 29 / 74 / 160 | 0 / 11 / 29 / 64 / 177 | 0 / 14 / 34 / 72 / 192 |
| T >= 1 / >= 32 / >= 64 / >= 128, feeds (share of qualifying) | 2,358 (72.9%) / 1,202 (37.2%) / 738 (22.8%) / **197 (6.1%)** | 1,383 (64.0%) / 691 (32.0%) / 367 (17.0%) / **155 (7.2%)** | 2,427 (59.2%) / 1,356 (33.1%) / 745 (18.2%) / 291 (7.1%) |
| U events per feed (A), 5 / 25 / 50 / 75 / 95% | 5 / 32 / 74 / 128 / 253 | 32 / 85 / 142 / 208 / 394 | 50 / 106 / 158 / 217 / 396 |
| U >= 1 / >= 32 / >= 64 / >= 128, feeds (share of qualifying) | 3,123 (96.5%) / 2,363 (73.0%) / 1,747 (54.0%) / 791 (24.5%) | 1,878 (86.9%) / 1,785 (82.6%) / 1,587 (73.5%) / 1,051 (48.7%) | 3,222 (78.5%) / 3,155 (76.9%) / 2,935 (71.6%) / 2,099 (51.2%) |
| share of views (type 1): stored events 129-256 / T / U | 0.442 / 0.307 / 0.450 | 0.230 / 0.293 / 0.632 | 0.225 / 0.294 / 0.646 |

The 14.3% on record is verified: 14.35% of test feeds (public-like days 15.03%, private-like 14.05%; the review's 14.00% counted train and val snapshots only, here earlier test snapshots are admitted too). Three facts the brief did not have:

1. **Only 10.8% of the holdout qualifies** (3,235 feeds, not a large share of the 11,600 capped ones): a holdout feed's user rarely has an earlier train feed. The review's power estimate ("full recovery about 6 SE on the holdout") does not hold.
2. **Few events are older than the stored 256th**: median 29; 128 of them exist for 6-7% of qualifying feeds (197 holdout, 155 val). The rule's pass (c), "128 events from beyond the cap", can be built as written for those feeds only.
3. **Most of what earlier snapshots add is not "beyond the 256th event"** but views removed by the per-type cap (U is 63-65% views on val / test; stored events 129-256 are 23% views there). For val / test this comes largely from the train-format snapshots, which keep twice as many views.

**Positive check of the method** (pretend the cap is 192: the stored history and every earlier snapshot cut to their newest 192; recover the events after the 192nd; truth = stored events 193-256) [M]:

| | holdout | val |
|---|---|---|
| feeds with >= 1 recovered event (with all 64) | 2,544 (752) | 1,417 (606) |
| recovered events compared | 91,870 | 61,462 |
| exact match at the same position | 41.9% | 28.5% |
| is one of the true events 193-256 (any position) | 72.3% | 60.4% |
| is stored among the newest 192 (a placement error of the method) | **0.4%** | **0.5%** |
| is not stored anywhere in the 256 (a real earlier event the later snapshot dropped) | 27.3% | 39.1% (74% of them views) |
| feeds with 64 recovered: share of the 64 true events found | 69.1% | 52.9% |
| same check with earlier snapshots at their stored length: exact position / true event | 48.5% / 75.8% | 33.4% / 66.4% |

Reading: the alignment does not misplace events (0.4-0.5%). The low exact-match rate is a property of the data: an earlier snapshot is a different sample of the user's event stream, so 27-39% of what it supplies beyond a cut are events the later snapshot would not store at any depth, and it misses 31-47% of the events the later snapshot does store there. A "refilled" history is therefore never the history a deeper cap would have stored.

## 3. Passes (step 2; `s2_replay.py`, `s3_read.py`, `s3c_pooled.py`, `s3d_table.py`; outputs `out_s2.txt`, `out_s3.txt`, `out_s3c.txt`, `out_s3d.txt`)

Read sets: design 1 (T) = feeds with criteria A and B and >= 1 event older than the stored 256th: holdout 2,358 (mean fill 46.7 events, 197 with all 128), val 1,383 (46.0; 155). Design 2 (U) = feeds with criterion A and >= 1 old-only event: holdout 3,123 (mean fill 73.8; 791 with all 128), val 1,878 (105.4; 1,051). Every feed has a placebo donor.

### 3.1 Mean NDCG of each pass, 4-pool and families [M]

Design 1 (T):

| pass | holdout pool4 | holdout distilled | holdout hard | val pool4 | val distilled | val hard |
|---|---|---|---|---|---|---|
| a: stored 256 | 0.31640 | 0.31601 | 0.31458 | 0.38037 | 0.38166 | 0.37993 |
| b: newest 128 | 0.31378 | 0.31389 | 0.31139 | 0.37770 | 0.37809 | 0.37772 |
| c: newest 128 + recovered | 0.31495 | 0.31549 | 0.31317 | 0.38039 | 0.37995 | 0.37973 |
| am: stored 128 + n | 0.31536 | 0.31566 | 0.31297 | 0.38071 | 0.38064 | 0.37854 |
| e: placebo (another user's events) | 0.31280 | 0.31371 | 0.31190 | 0.37654 | 0.37449 | 0.37673 |
| b192: newest 192 | 0.31437 | 0.31477 | 0.31322 | 0.38129 | 0.37988 | 0.38079 |
| c192: newest 192 + recovered | 0.31604 | 0.31607 | 0.31539 | 0.38237 | 0.38189 | 0.38422 |
| g: newest 128 + stored 193-256 | 0.31495 | 0.31494 | 0.31367 | 0.37917 | 0.37978 | 0.37772 |
| d1: 256 + recovered, dense inputs held | 0.31669 | 0.31586 | 0.31579 | 0.38222 | 0.38258 | 0.38000 |
| d2: 256 + recovered, dense inputs recomputed | 0.31737 | 0.31618 | 0.31673 | 0.38145 | 0.38308 | 0.38078 |
| dP: 256 + another user's events | 0.31558 | 0.31365 | 0.31439 | 0.37947 | 0.38026 | 0.38017 |

Design 2 (U):

| pass | holdout pool4 | holdout distilled | holdout hard | val pool4 | val distilled | val hard |
|---|---|---|---|---|---|---|
| a: stored 256 | 0.31902 | 0.31814 | 0.31689 | 0.38803 | 0.38855 | 0.38573 |
| b: newest 128 | 0.31664 | 0.31638 | 0.31483 | 0.38397 | 0.38377 | 0.38186 |
| c: newest 128 + recovered | 0.31814 | 0.31829 | 0.31649 | 0.38665 | 0.38803 | 0.38436 |
| am: stored 128 + n | 0.31834 | 0.31788 | 0.31575 | 0.38663 | 0.38788 | 0.38439 |
| e: placebo (another user's events) | 0.31438 | 0.31379 | 0.31278 | 0.38478 | 0.38475 | 0.38304 |
| b192: newest 192 | 0.31735 | 0.31688 | 0.31634 | 0.38734 | 0.38713 | 0.38467 |
| c192: newest 192 + recovered | 0.31767 | 0.31779 | 0.31727 | 0.38605 | 0.38797 | 0.38664 |
| g: newest 128 + stored 193-256 | 0.31761 | 0.31735 | 0.31666 | 0.38604 | 0.38620 | 0.38446 |
| d1: 256 + recovered, dense inputs held | 0.31939 | 0.31802 | 0.31819 | 0.38739 | 0.38737 | 0.38693 |
| d2: 256 + recovered, dense inputs recomputed | 0.31913 | 0.31761 | 0.31800 | 0.38844 | 0.38639 | 0.38762 |
| dP: 256 + another user's events | 0.31650 | 0.31518 | 0.31642 | 0.39024 | 0.38801 | 0.38593 |

### 3.2 Paired contrasts, 4-pool [M] (SE cluster-robust by user; "pooled" = feeds of both read sets)

| contrast | design 1 holdout (2,358) | design 1 val (1,383) | design 1 pooled (3,741) | design 2 holdout (3,123) | design 2 val (1,878) | design 2 pooled (5,001) |
|---|---|---|---|---|---|---|
| a - b: worth of the true events 129-256 (denominator) | +0.00262 +- 0.00154 | +0.00267 +- 0.00247 | +0.00264 +- 0.00133 | +0.00238 +- 0.00130 | +0.00405 +- 0.00207 | +0.00301 +- 0.00113 |
| **c - b: worth of the recovered events (numerator)** | **+0.00117 +- 0.00082 (z 1.43)** | +0.00269 +- 0.00157 (z 1.71) | +0.00173 +- 0.00078 (z 2.23) | **+0.00151 +- 0.00093 (z 1.62)** | +0.00267 +- 0.00215 (z 1.24) | +0.00194 +- 0.00099 (z 1.96) |
| am - b: stored events in the same slots | +0.00158 +- 0.00100 | +0.00301 +- 0.00183 | +0.00211 +- 0.00092 | +0.00170 +- 0.00099 | +0.00265 +- 0.00183 | +0.00206 +- 0.00092 |
| c - am: recovered minus stored, same slots | -0.00041 +- 0.00105 | -0.00032 +- 0.00183 | -0.00038 +- 0.00095 | -0.00020 +- 0.00103 | +0.00002 +- 0.00200 | -0.00012 +- 0.00099 |
| e - b: placebo | -0.00098 +- 0.00077 | -0.00116 +- 0.00129 | -0.00105 +- 0.00068 | -0.00226 +- 0.00087 (z -2.60) | +0.00081 +- 0.00174 | -0.00111 +- 0.00085 |
| c - e: own old events minus another user's | +0.00215 +- 0.00093 (z 2.31) | +0.00385 +- 0.00194 (z 1.98) | +0.00278 +- 0.00093 (z 2.99) | +0.00376 +- 0.00112 (z 3.36) | +0.00187 +- 0.00237 (z 0.79) | +0.00305 +- 0.00113 (z 2.70) |
| d1 - a: 256 + recovered, untrained for it | +0.00029 +- 0.00058 | +0.00186 +- 0.00132 | +0.00087 +- 0.00061 | +0.00037 +- 0.00071 | -0.00064 +- 0.00172 | -0.00001 +- 0.00078 |
| d2 - a | +0.00097 +- 0.00065 | +0.00108 +- 0.00147 | +0.00101 +- 0.00068 | +0.00011 +- 0.00073 | +0.00041 +- 0.00178 | +0.00022 +- 0.00081 |
| dP - a: 256 + another user's events | -0.00082 +- 0.00074 | -0.00089 +- 0.00120 | -0.00084 +- 0.00065 | -0.00252 +- 0.00077 (z -3.26) | +0.00221 +- 0.00150 | -0.00074 +- 0.00074 |
| d1 - dP | +0.00111 +- 0.00085 | +0.00275 +- 0.00172 | +0.00172 +- 0.00083 (z 2.06) | +0.00289 +- 0.00091 (z 3.19) | -0.00285 +- 0.00198 | +0.00074 +- 0.00094 |

### 3.3 By the number of slots filled, c - b (and a - b), 4-pool [M]

| filled | design 1 holdout | design 1 val | design 1 pooled | design 2 holdout | design 2 val | design 2 pooled |
|---|---|---|---|---|---|---|
| 1-31 | +0.00155 +- 0.00064 (n 1,156; a - b +0.00223 +- 0.00199) | +0.00004 +- 0.00109 (n 692) | +0.00098 +- 0.00057 | -0.00064 +- 0.00088 (n 760) | +0.00334 +- 0.00683 (n 93) | -0.00021 +- 0.00108 |
| 32-63 | -0.00070 +- 0.00242 (n 464; +0.00061 +- 0.00491) | +0.00279 +- 0.00364 (n 324) | +0.00073 +- 0.00207 | -0.00056 +- 0.00218 (n 616) | -0.00306 +- 0.00382 (n 198) | -0.00117 +- 0.00189 |
| 64-127 | +0.00099 +- 0.00217 (n 541; +0.00562 +- 0.00267) | +0.00704 +- 0.00659 (n 212) | +0.00269 +- 0.00242 | +0.00251 +- 0.00176 (n 956; +0.00413 +- 0.00240) | +0.00079 +- 0.00441 (n 536) | +0.00189 +- 0.00194 |
| 128 (pass c as the rule words it) | +0.00387 +- 0.00367 (n 197; +0.00144 +- 0.00403) | +0.00837 +- 0.00578 (n 155; +0.01025 +- 0.00710) | +0.00585 +- 0.00327 (z 1.79) | +0.00396 +- 0.00230 (n 791; +0.00272 +- 0.00186) | +0.00466 +- 0.00297 (n 1,051; +0.00354 +- 0.00278) | +0.00436 +- 0.00196 (z 2.22) |

The gain grows with the number of events filled in both designs (pooled: +0.0010 / +0.0007 / +0.0027 / +0.0059 and -0.0002 / -0.0012 / +0.0019 / +0.0044); no single cell is 2 SE from zero on one set.

### 3.4 What the passes say, in words

- **The recovered events are worth about what the stored events in the same slots are.** c - am is -0.0004 +- 0.0010 (design 1, pooled) and -0.0001 +- 0.0010 (design 2, pooled); nothing separates them on either set.
- **It is the user's own past that carries it, not "more events".** Another user's events in the same slots read below the empty slots (e - b -0.0011 +- 0.0007 and -0.0011 +- 0.0009 pooled, 1.3-1.5 SE) and clearly below the user's own (c - e +0.0028 +- 0.0009, z 3.0; +0.0031 +- 0.0011, z 2.7). On val alone the placebo of design 2 is not separated (c - e +0.0019 +- 0.0024).
- **Every member agrees in sign.** The numerator c - b is positive for 4 of 4 members on both sets in both designs (16 of 16 cells; design 1 holdout +0.0018, +0.0017, +0.0006, +0.0014; val +0.0028, +0.0022, +0.0011, +0.0035; design 2 holdout +0.0011, +0.0024, +0.0004, +0.0017; val +0.0037, +0.0025, +0.0035, +0.0040); single cells are 0.4-2.4 SE.
- **A model not trained for it gains nothing reliable from a 384-event history.** d1 - a: +0.0009 +- 0.0006 (design 1, pooled), 0.0000 +- 0.0008 (design 2, pooled). The foreign-event form is below it in design 1 and on the holdout in design 2; on val in design 2 the placebo reads higher than the real events (-0.0029 +- 0.0020), i.e. noise of that size is present.
- **Calibration of the replay design (g).** Stored events 193-256 shown in slots 129-192 return rho_g = 1.23 [0.64, 2.69] of the worth of the true events 129-192 on all 15,777 capped feeds of both sets (holdout 1.78 [0.95, 5.94]; val 0.15 [-2.31, 2.38]): showing events in slots 64 positions too shallow does not destroy their worth, so the slot substitution of pass (c) is a fair instrument. On the holdout the marginal 64-event stretches read +0.00141 +- 0.00053 (events 129-192) and +0.00174 +- 0.00050 (events 193-256), 4-pool, all capped feeds: the review's "no decay with depth" holds there too.
- **Sanity against the review.** Truncation cost on all capped feeds, hard-label members (mean of the two member deltas), val: K = 192 -0.00247 +- 0.00093 (review -0.00246 +- 0.00093), K = 128 -0.00456 +- 0.00123 (review -0.00475 +- 0.00122). My GPU pass and the review's CPU file correlate 0.9999998; member-level costs differ by about 0.0002 because the top-16 retrieval breaks ties between repeated items differently on CPU and GPU (8-9% of capped feeds change NDCG between the two devices). New [M]: the truncation cost holds on the holdout, 4-pool, 11,601 capped feeds: K = 192 -0.00174 +- 0.00050 (z -3.5), K = 128 -0.00314 +- 0.00066 (z -4.7).

## 4. Read by the declared rule (step 3)

rho = [N(c) - N(b)] / [N(a) - N(b)], 4-pool, paired per feed, user-cluster SE, 2,000-draw user bootstrap for the interval. OPEN iff rho >= 0.5 with the numerator at z >= 2 on the holdout and the same sign on val; CLOSE iff the numerator's upper 95% bound (mean + 1.96 SE) is below 25% of the denominator. All [M].

### 4.1 The verdict row and the same row of the second design

| | n (holdout) | numerator c - b | denominator a - b | rho [95%] | upper bound of the numerator vs 25% of the denominator | val numerator (n) | rule |
|---|---|---|---|---|---|---|---|
| **design 1 (T), literal: the verdict row** | 2,358 | +0.00117 +- 0.00082 (z 1.43) | +0.00262 +- 0.00154 (z 1.70) | **0.447** [-1.40, +2.80] | +0.00277 vs +0.00065 | +0.00269 +- 0.00157 (1,383) | **UNDECIDED** |
| design 2 (U), literal | 3,123 | +0.00151 +- 0.00093 (z 1.62) | +0.00238 +- 0.00130 (z 1.83) | 0.632 [-1.21, +3.40] | +0.00332 vs +0.00060 | +0.00267 +- 0.00215 (1,878) | UNDECIDED |

Neither condition is met in either design: no numerator reaches z = 2 on the holdout, and no upper bound comes near a quarter of the denominator. With the one-sided 1.645 form the upper bounds are +0.00251 and +0.00303: the same.

**Why the rule could not decide on this sample** [D]. The denominator itself (the worth of the true events 129-256) is only 1.7-1.8 SE on the holdout read sets. With the measured SEs, OPEN required rho >= 0.63 (design 1) or 0.78 (design 2) and CLOSE required rho <= -0.36 or -0.52: only a recovery clearly above one half, or old events that hurt, could have been called. The review's "full recovery about 6 SE on the holdout" assumed several times more qualifying feeds than exist (3,235).

### 4.2 The declared companion reads (same thresholds, not verdicts)

| read | holdout numerator | holdout denominator | rho [95%] | val numerator | thresholds say |
|---|---|---|---|---|---|
| design 1, full fill (n = 128; pass c exactly as the rule words it) | +0.00387 +- 0.00367 (n 197) | +0.00144 +- 0.00403 | 2.68 [-13.4, +15.5] | +0.00837 +- 0.00578 (n 155) | UNDECIDED |
| design 1, length-matched (denominator am - b) | +0.00117 +- 0.00082 | +0.00158 +- 0.00100 | 0.74 [-2.70, +5.13] | +0.00269 +- 0.00157 | UNDECIDED |
| design 1, ceiling [am - b] / [a - b] (what perfect events of the same count would give) | | | 0.60 [-1.55, +2.55]; val 1.13 | | |
| design 2, full fill (n = 128) | +0.00396 +- 0.00230 (n 791) | +0.00272 +- 0.00186 | 1.46 [-5.93, +9.60] | +0.00466 +- 0.00297 (n 1,051) | UNDECIDED |
| design 2, length-matched | +0.00151 +- 0.00093 | +0.00170 +- 0.00099 | 0.88 [-1.49, +4.98] | +0.00267 +- 0.00215 | UNDECIDED |
| design 2, ceiling | | | 0.71 [-0.65, +2.97]; val 0.65 [-0.37, +1.47] | | |
| pooled holdout + val, design 1 literal (n 3,741) | +0.00173 +- 0.00078 (z 2.23) | +0.00264 +- 0.00133 | 0.66 [-0.13, +2.94] | | descriptive |
| pooled holdout + val, design 2 literal (n 5,001) | +0.00194 +- 0.00099 (z 1.96) | +0.00301 +- 0.00113 | 0.65 [+0.01, +1.81] | | descriptive |
| pooled, design 2 full fill (n 1,842) | +0.00436 +- 0.00196 (z 2.22) | +0.00319 +- 0.00177 | 1.37 [-2.47, +6.31] | | descriptive |

The pooled rows use every labelled feed on which these members are out of sample (holdout and val read sets): there is no larger sample to replay on. They put the numerator at about 2 SE and rho near 0.65, with an interval that still includes zero in design 1.

### 4.3 By member family

| family | design | holdout numerator | holdout denominator | rho [95%] | val numerator | thresholds say | pooled numerator, rho [95%] |
|---|---|---|---|---|---|---|---|
| distilled (2 students) | 1 | +0.00159 +- 0.00087 (z 1.84) | +0.00212 +- 0.00152 | 0.75 [-4.16, +5.65] | +0.00186 +- 0.00172 | UNDECIDED | +0.00169 +- 0.00084 (z 2.02), 0.64 [-0.23, +2.97] |
| hard-label (2 members) | 1 | +0.00177 +- 0.00084 (z 2.10) | +0.00319 +- 0.00148 | 0.56 [-0.08, +2.92] | +0.00200 +- 0.00164 | OPEN (sub-row) | +0.00186 +- 0.00081 (z 2.30), 0.66 [-0.02, +3.06] |
| distilled | 2 | +0.00191 +- 0.00090 (z 2.12) | +0.00176 +- 0.00132 | 1.09 [-6.48, +6.92] | +0.00426 +- 0.00212 (z 2.00) | OPEN (sub-row) | +0.00279 +- 0.00097 (z 2.86), 0.97 [+0.35, +3.06] |
| hard-label | 2 | +0.00166 +- 0.00093 (z 1.79) | +0.00206 +- 0.00127 | 0.81 [-4.59, +5.13] | +0.00250 +- 0.00203 | UNDECIDED | +0.00197 +- 0.00096 (z 2.07), 0.72 [+0.06, +2.50] |

The two families do not differ: numerators +0.0016..+0.0019 on the holdout in all four cells, z 1.8-2.1. Two of the four cells cross the OPEN thresholds; with 36 threshold-labelled rows in `out_s3.txt` (6 read OPEN, 30 UNDECIDED, 0 CLOSE) that is not a verdict. The 4-pool numerators are a little below both families' (+0.0012 and +0.0015): pooling does not average NDCG deltas.

### 4.4 Placebo, second line, direct pass

- **Placebo (e in place of c)**: rho_e = -0.38 [-3.41, +2.94] (design 1, holdout), -0.44 (val); -0.95 [-7.67, +3.79] (design 2, holdout), +0.20 [-1.35, +2.10] (val). "Any extra events help" is not what the numerator measures: foreign events read at or below the empty slots, and the user's own events beat them by +0.0028 +- 0.0009 (design 1) and +0.0031 +- 0.0011 (design 2), pooled.
- **192 + 64 line** (newest 192 + up to 64 recovered events, against the newest 192): design 1 holdout +0.00167 +- 0.00078 (z 2.13) on a denominator a - b192 of +0.00203 +- 0.00117 (rho 0.82 [-1.77, +5.03]); val +0.00108 +- 0.00159 on a negative denominator (-0.00093 +- 0.00186: no ratio). Design 2: +0.00032 +- 0.00072 (holdout), -0.00130 +- 0.00161 (val). The line carries no decision: its denominators are within 1.7 SE of zero on these read sets, and in design 2 the newest old-only events (mostly views removed by the per-type cap) add nothing on top of the newest 192.
- **Pass (d) minus (a)**, the direct estimate for a model not trained for it: design 1 +0.00029 +- 0.00058 (holdout), +0.00186 +- 0.00132 (val), pooled +0.00087 +- 0.00061; design 2 +0.00037 +- 0.00071, -0.00064 +- 0.00172, pooled -0.00001 +- 0.00078. With the dense inputs recomputed (d2): +0.00101 +- 0.00068 and +0.00022 +- 0.00081 pooled. By family, pooled d1 - a: distilled +0.00025 +- 0.00074 / -0.00051 +- 0.00095; hard-label +0.00079 +- 0.00060 / +0.00126 +- 0.00083. Not distinguishable from zero.

## 5. Size at the pool level of the final file (step 4)

Shares are label-free [M] (alignment run on val and test, `out_s1.txt`); effects are the 4-pool contrasts of section 3, estimated per fill bucket (1-31 / 32-63 / 64-127 / 128 events) on the holdout and val read sets together and weighted by the segment's bucket shares [D]. Two effects bracket the answer: **N(c) - N(b)**, the worth of the recovered events when shown in slots the model was trained on (an upper-side proxy for a retrained model: it assumes a retrained model uses events at depth 257-384 as well as today's models use events at depth 129-256, and that nothing is lost to train / test differences), and **N(d1) - N(a)**, what today's models get from the extended history without retraining.

| segment | qualifying feeds | design 1: feeds that receive events | design 1: c - b proxy | design 1: d1 - a direct | design 2: feeds that receive events | design 2: c - b proxy | design 2: d1 - a direct |
|---|---|---|---|---|---|---|---|
| val | 14.98% | 9.59% | +0.00017 +- 0.00007 | +0.00008 +- 0.00006 | 13.02% | +0.00037 +- 0.00016 | +0.00005 +- 0.00013 |
| test, public-like days | 15.03% | 9.29% | +0.00017 +- 0.00008 | +0.00009 +- 0.00006 | 12.21% | +0.00040 +- 0.00017 | +0.00007 +- 0.00014 |
| test, private-like days | 14.05% | 8.15% | +0.00015 +- 0.00007 [95% +0.00001, +0.00028] | +0.00007 +- 0.00005 [-0.00003, +0.00017] | 10.87% | **+0.00035 +- 0.00015 [95% +0.00006, +0.00064]** | **+0.00006 +- 0.00012 [-0.00018, +0.00029]** |
| test, all | 14.35% | 8.49% | +0.00015 +- 0.00007 | +0.00008 +- 0.00005 | 11.27% | +0.00037 +- 0.00016 | +0.00006 +- 0.00012 |

Sensitivity [D]: with effects from the holdout only the private-like sizes are +0.00009 +- 0.00007 (design 1) and +0.00034 +- 0.00017 (design 2); from val only +0.00025 +- 0.00014 and +0.00033 +- 0.00025. The plain form asked for (mean effect over ALL qualifying feeds, zeros where nothing is recovered, times the 14.05% of private-like days): design 1 +0.00012 +- 0.00008 (holdout effect) / +0.00024 +- 0.00014 (val effect); design 2 +0.00020 +- 0.00013 / +0.00033 +- 0.00026.

Reading: **what is reachable is +0.0002 (events older than the 256th only) to +0.0004 (all old-only events) at pool level on private-like days if a retrained model captures the full proxy, and about +0.0001 +- 0.0001 without retraining.** The review's "+0.0002..+0.0005 if events beyond the cap are worth what events 193-256 are" stands as an upper range; its lower half is the realistic part, because two thirds of the recoverable events are not older than the 256th event and on top of 192 stored events they read zero (192 + 64 line of design 2: +0.0003 +- 0.0007 holdout, -0.0013 +- 0.0016 val). These numbers are for the train-only pool; the fine-tuned half of the final file was not replayed.

## 6. The arm (step 5)

**The rule does not say OPEN, so no arm is specified as a decision.** It does not say CLOSE either. What follows is the record the lead needs if the door is kept open on the pooled evidence; it is not a recommendation to launch.

**What an extended-history arm would have to be** (all label-free parts were measured here, `out_s1.txt`, `out_parity.txt`):

1. *Tables* (CPU, about 10 minutes with `align_lib.py`): for every feed of train / val / test with H = 256, user_id != 0 and a strictly earlier snapshot (train feeds: earlier train feeds only; val: train + earlier val; test: train + val + earlier test), append the first min(n, 128) old-only events of the union timeline (design 2 order) after the stored 256; feeds failing criterion A stay as stored. New history arrays (`hist_emb_idx`, `hist_category_id`, `hist_event_type`, `hist_event_source`) and feed-level `H` / `hist_off` per split; for val / test the item table needs the rows of the train table that recovered events point to (the same item has an identical vector in 80-87% of cases, cosine >= 0.985 for 95% of the rest).
2. *Coverage and parity* [M]: feeds that receive events: holdout 10.4%, val 13.0%, test 11.3% (private-like 10.9%); qualifying train feeds 11.6% (1.9% on the first train day, 18.8% on day 13). The dose is not the same: mean events appended per extended feed 74 on the holdout against 105 on val; share of views among them 45% against 63-65%; median gap to the earlier snapshot 2.9 days against 9.6 (val) and 14.2 (test); alignment failures 2.9% against 13.0% and 21.5%. Train rows cannot be built "exactly like test rows" (rule of section 3 of CURRENT_STATUS): the train period is 14 days, the test gap alone is 14 days. The nearest fix is to cut the stored train histories to the val / test per-type caps first (the existing `--hcap "1:110,4:117,6:41,8:42,10:40"`, itself a null arm) so that the appended events have the val / test mix; the gap cannot be matched.
3. *Flags that exist*: `--max_hist` in `train.py` / `train_full2.py`; `finetune.py` reads it from the checkpoint; `SplitData(max_hist=...)` pads to multiples of 64. *What must be coded*: (i) `model.py`: `rec = nn.Embedding(256, d)` is hard-coded, so a length key in cfg (stored only when not 256, as the r6 options are) with rows 256-383 initialised from row 255; the retrieval recency bucket already saturates at 8; (ii) `data.py`: a history-table directory argument (the module reads `data/flat/<split>/hist_*.npy` and one item table per split); (iii) the table builder; (iv) a flags-off identity test. The 28 history-dependent precomputed inputs (`seq_cv_*`, `ctx_*`, `pre2_*`) would stay on the stored 256 events unless rebuilt.
4. *Cost* [D / I]: a kd05w192 student takes 77-245 minutes (stored logs, as read by the review); longer sequences touch about 12% of train feeds, so 4 students are about 6-18 GPU-hours and 16 students 23-72, plus the n2cpv3 fine-tunes (5 folds per member; about 1-2 GPU-hours per 4 members [I]). A cheaper pilot exists: re-cool the stored `@77500` plateau snapshots with the extended tables (the mechanism of the hz re-cools, 10-19 minutes each), about 1 GPU-hour for 4 and 3-5 for 16; it tests whether the cooldown alone can teach positions 257-384 [I].
5. *Kill rule I would declare* (non-staleness family, so val AND holdout): read on the bucket of extended feeds (val 1,879, holdout 3,140), 4 students against their matched kd05w192 seeds. Expected bucket delta if the whole proxy carries over: +0.0027 (val) and +0.0015 (holdout); A/A SE of a 4-pool contrast on buckets of this size about 0.0016 and 0.0010 [D from the review's A/A null]. KILL after 4 students if the pooled bucket z is below +1, or if the overall val or holdout delta is below -1 SE; extend to 8 only at pooled bucket z >= 1.6 with both bucket deltas >= 0; adopt only at pooled bucket z >= 2 with the overall deltas not negative. Normal-approximation power [D]: first gate about 0.9 at the full proxy effect and 0.55 at half of it; adoption bar with 8 students about 0.75 at the full proxy effect and 0.3 at half of it.

**My reading for the lead** [I]: the mechanism is more likely real than not (pooled numerator about 2 SE, own-versus-foreign 2.7-3.0 SE, 16 of 16 member cells positive, recovered events equal to stored ones within +-0.001), but the prize is +0.0002..+0.0004 at pool level only if retraining transfers the whole proxy across a train / test dose mismatch, and today's models get +0.0001 +- 0.0001 from the extension. Probability of a pool gain >= +0.0003 from a 16-student arm: about 0.15; >= +0.0005: about 0.03 (the review's 0.15 / 0.05 before the replay: the mechanism moved up, the size moved down). By the letter of the rule the second design is used up and the door closes without an arm; what that leaves on the table is at most the +0.0002..+0.0004 above. If idle GPU-hours are to be spent on it at all, only the 4-snapshot re-cool pilot with the kill rule above is proportionate.

## 7. Limits, deviations, hard rules, files

**Limits of the read.**
- The replay shows recovered events in slots trained for other depths (pass c) or beyond every trained position (pass d). It measures whether the events carry model-usable information, not what a retrained model would gain. Calibration g says the slot substitution itself is harmless (rho_g 1.23 [0.64, 2.69]).
- Up to 28 of the 34 precomputed inputs and the organisers' user embedding stay on the stored 256 events in every pass: all contrasts are lower bounds in magnitude, numerator and denominator alike.
- The decision sample is in train format (per-type caps twice as deep in half of the capped feeds, earlier snapshot 2.9 days back); val is the test-like sample (9.6 days; test 14.2) and is half as large. Val numerators are larger than holdout ones in both designs (+0.0027 against +0.0012 / +0.0015), inside their SE.
- CPU and GPU runs of the same pass differ in NDCG on 8-9% of capped feeds (tie order of the top-16 retrieval among repeated items); all passes here ran on one device with one code path, so paired contrasts are not affected in expectation.
- 36 rows of `out_s3.txt` carry a threshold label (6 OPEN, 30 UNDECIDED, 0 CLOSE); only the row declared in block 0 is the verdict. Family, bucket, pooled and length-matched rows are companions.

**Deviations from the brief, all declared before the first labelled number unless marked.**
- Two recovery definitions (T, U) instead of one, because the label-free step showed that "the 128 events beyond the cap" exist for 6-7% of qualifying feeds only. The verdict row stayed the literal one (T).
- Pool = member-equal mean of per-feed score z (the project's `feed_z`), not rank z as the brief words it.
- Pass (d) needed a convention (`rec` index held at 255 beyond position 255); the brief assumed no positional embedding.
- "Exact-match rate" of the positive check is reported by position and by membership, with the decomposition that explains why the positional rate is low.
- Flock on the lane lock with a 30-minute limit instead of an unbounded wait (obtained at once).
- AFTER the read (blocks 2-3 of the declaration): `s3b` integrity checks, `s3c_pooled.py` (pooled forms of contrasts printed per set only), `s3d_table.py` (levels), `out_parity.txt` (label-free). No statistic of block 0 was changed or re-run; the rule was read once.
- Not done: the fine-tuned branch was not replayed (train-only checkpoints only); the 192 + 64 line is reported but carries no decision.

**Hard rules.** GPU: one process (PID [n]), GPU N only, inference only, 9.8 minutes, launched under the lane lock with 3 compute processes and 88,742 MiB free, lock held until the process was visible; nothing else touched a GPU. CPU: one process at a time, 12 threads, resident memory a few GB (all train arrays memory-mapped, only the needed rows read). No process signalled. Writes only in `nn_x/w12/seg/refill/` and this report (the lane lock file under `nn_x/w5/logs/locks/` was opened as the lane protocol prescribes; no bytecode written into `nn_x/w5`). No network, no install, no submission file. Leak rules: only snapshots with a strictly smaller timestamp; user_id 0 excluded; no label in any input (labels are read by `s3_read.py` and its addenda only, on finished score files); members held-out (`args.holdout = [n]`).

**Files** (`nn_x/w12/seg/refill/`): `DECLARATION.txt`, `PROGRESS.md`; `align_lib.py` (LCS alignment, union timeline), `s1_align.py` -> `out_s1.txt`, `out/align_{holdout,val,test}.npz`; `s1b_poscheck.py` -> `out_s1b.txt`; `s2_replay.py` + `run_gpu7.sh` -> `out_s2.txt`, `out/scores_<set>_<member>.npz` (20 passes each), `out/meta_<set>.npz`; `s3_read.py` -> `out_s3.txt`, `out/s3_results.json`; `s3c_pooled.py` -> `out_s3c.txt`; `s3d_table.py` -> `out_s3d.txt`; `out_s3b_checks.txt`; `out_parity.txt`; report parts `rep_*.txt` and `build_report.py`.
