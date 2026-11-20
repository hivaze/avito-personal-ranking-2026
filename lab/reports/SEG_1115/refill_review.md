# SEG_1115 / refill_review: adversarial review of the refill replay (`refill.md`)

Written 2026-11-15T18:43Z. Reviewer directory `nn_x/w12/seg/refill_review/` (scripts, raw outputs `out_*.txt`, `DECLARATION.txt` with three stamped blocks, `PROGRESS.md`). Rule under review: `nn_x/w5/logs/E_kd_predeclared.txt`, block 2026-11-15T17:41:25Z, unchanged. Everything below was computed with my own code (own aligner, own subset loader and batching, own NDCG, own cluster SE, own bootstrap); nothing of `nn_x/w12/seg/refill/*.py` was imported; the study's files were read as data only. `nn_x/w5/model.py` was imported read-only. Marks: [M] measured here, [D] arithmetic from [M], [I] judgement. SE are cluster-robust by user. My declaration (block 0, 18:26:36Z; block 1, 18:32:08Z) precedes my first labelled number (18:37:33Z); the labelled read was run once.

## 0. Verdict

**confirmed = true.** The study's verdict, UNDECIDED by the declared rule, stands. It was re-derived three ways and no leak, no implementation error and no design artefact was found.

1. **The verdict row is reproduced exactly** [M]. Study's recovered events, my inference and my statistics (design 1, 4-pool, holdout, 2,358 feeds): numerator N(c) - N(b) = +0.00117 +- 0.00082 (z 1.43), denominator +0.00262 +- 0.00154, rho 0.447; val numerator +0.00269 +- 0.00157. Not OPEN (z below 2, rho below 0.5), not CLOSE (upper bound +0.00277 against +0.00065). My scores equal the study's stored scores to 3e-5 on the holdout and 3e-4 on val for every pass I re-ran (a, b, c, am; all four members).
2. **With my own aligner the same row reads the same** [M]: read set 2,361 feeds, numerator +0.00120 +- 0.00084 (z 1.43), denominator +0.00255 +- 0.00154, rho 0.472; val +0.00300 +- 0.00152: UNDECIDED. Design 2: study's events +0.00151 +- 0.00093 (z 1.62, rho 0.63); mine +0.00164 +- 0.00095 (z 1.73, rho 0.72): UNDECIDED.
3. **No leak** [M]. All 1,107,597 recovered rows of the study (T and U; holdout, val, test) lie inside the history of a feed of the same user_id with a strictly smaller timestamp (smallest gap 128 s, one timestamp step); no user_id 0 feed is in any list; holdout rows come from the train table only.
4. **No design artefact** [M]. Length inputs alone give +0.0002 +- 0.0004; dropping the recovered events that would merely restore items of the removed stored events does not lower the numerator (+0.00196 +- 0.00077 and +0.00205 +- 0.00093 pooled, against +0.00173 and +0.00194); the user's own events beat another user's by +0.0027 +- 0.0009 (z 2.95) and +0.0027 +- 0.0011 (z 2.40) with my own donor draw.
5. **What the lead should know beyond the label** [M / D]: the rule could not have returned OPEN on this sample for any events. The stored events themselves, placed in the same slots, read +0.00158 +- 0.00100 (z 1.58) and +0.00170 +- 0.00099 (z 1.73) on the holdout. UNDECIDED is a power failure, not evidence against the mechanism; every point estimate sits on the OPEN side (rho 0.44-0.81, z 1.4-2.2), none near CLOSE.
6. **Size** [D]: +0.00014 +- 0.00006 (events older than the 256th) to +0.00021 +- 0.00011 (all old-only events) at train-only pool level on private-like days in the plain form; the study's +0.00035 for design 2 is the bucketed form and rests on one bucket at z 2.2. Both assume a retrained model captures all of it.
7. **Recommendation** [I]: no arm. Record the door as closed for lack of a decidable test, with the mechanism noted as probably real and small.

## 1. What was checked and how

| task item | my code | outcome |
|---|---|---|
| 1. alignment, 300 random qualifying feeds per set | `rr_lib.py` (patience anchors + multiset gaps), `s1_checks.py`, `s1b_order_parity.py` | T events 99% identical to the study's; positive check reproduced |
| 2. leak | `s1_checks.py` section 1 (all rows, not 300) | none |
| 3. passes a, b, c, e with my batching, rho with my bootstrap | `s2_passes.py` (GPU N, lane lock), `s3_read.py` | verdict row reproduced exactly |
| 4. artefacts | passes bL, cN, cX and my own placebo draw | none material |
| 5. size, parity | `s3_read.py` section 4, `s1_checks.py` section 5, `s1b_order_parity.py` | size confirmed at its lower half; parity not reachable |

My rule for `confirmed`, written before any number (block 0): R1 no provenance violation; R2 verdict-row numerator and denominator reproduced within 0.0003 with the same label; R3 my own alignment does not read OPEN or CLOSE; R4 no control removes the sign or more than half of the numerator at 2 SE. All four hold.

Members: the study's four, held-out (`args.holdout == 30000` asserted from each checkpoint): distilled `w5_e10_kd05w192_s951`, `w5_e10_kd05t25w192_s951` (d192); hard-label `w5_e10_base_s951`, `w5_e10_base_s952` (d128). Pool = member-equal mean of per-feed score z. Inference fp32, one process on GPU N under the lane lock, 308 s. Fidelity of pass a against the stored predictions: flat correlation 0.999997-1.000000 on all 8 member x set.

## 2. Alignment and leak (items 1 and 2)

All label-free [M], `out_s1.txt`, `out_s1b.txt`.

**Qualifying feeds.** My lists are identical to the study's: holdout 3,235 (10.78%), val 2,160 (14.98%), test 4,102 (14.35%; public-like 15.03%, private-like 14.05%).

**Provenance of every recovered row** (through `hist_fidx`, raw tables):

| set | T rows | U rows | outside the feed's history | other user_id | snapshot not strictly earlier | smallest gap |
|---|---|---|---|---|---|---|
| holdout | 121,114 | 230,390 | 0 | 0 | 0 | 128 s |
| val | 67,880 | 197,967 | 0 | 0 | 0 | 640 s |
| test | 127,776 | 362,470 | 0 | 0 | 0 | 256 s |

Timestamps have a 128-second step; train ends before val starts and val before test. A recovered event cannot postdate the feed it extends: it is taken from the history of a snapshot whose own request time is earlier, and the alignment only decides where an event is placed, never which events exist. A mis-ordered alignment can misplace an event; it cannot import a later one.

**Independent aligner.** Mine uses keys that occur once in both sequences, made monotone by a longest increasing subsequence, and pairs equal keys inside the gaps as multisets; the study uses an LCS with tie-swap repair.

| | holdout | val | test |
|---|---|---|---|
| read set T: mine / study / common | 2,361 / 2,358 / 2,351 | 1,377 / 1,383 / 1,367 | 2,425 / 2,427 / 2,401 |
| read set U: mine / study / common | 3,122 / 3,123 / 3,113 | 1,868 / 1,878 / 1,867 | 3,193 / 3,222 / 3,193 |
| study's T events that are also mine (key multiset) | 99.3% | 99.0% | 98.8% |
| study's U events that are also mine | 97.6% | 96.5% | 96.8% |
| correlation of the T count per feed | 0.996 | 0.989 | 0.982 |
| T per feed, median: mine / study | 33 / 33 | 32 / 31 | 37 / 37 |

Stored order is newest first by my code as well (holdout: 37.4 stored-only events before the first anchor against 14.0 after the last; 27.2 earlier-only events after the snapshot's last anchor against 0.9 before its first).

**Are the T events older than the stored 256th?** Aligner-free test on my 300 random feeds per set (rng 4242): a T event taken from a snapshot must sit deeper in that snapshot than every event the snapshot shares with the stored history. Restricted to shared events stored at index 200 or deeper (so that a repeated key whose old occurrence left the window is not counted): 0.09% of T events fail on the holdout (1 feed of 218), 1.36% on val (3 of 197), 2.31% on test (7 of 175). Without that restriction 3.3% / 9.0% / 4.4%. Immaterial.

**Positive check with my aligner** (cap pretended 192, snapshots cut to 192, truth = stored events 193-256), my 300 feeds:

| | holdout mine (study, all feeds) | val mine (study) | test mine (study: not reported) |
|---|---|---|---|
| is a true event 193-256 | 72.8% (72.3%) | 60.5% (60.4%) | 53.7% |
| exact position | 38.6% (41.9%) | 31.7% (28.5%) | 21.8% |
| stored among the newest 192 (placement error) | 0.5% (0.4%) | 1.3% (0.5%) | 0.8% |
| not stored anywhere in the 256 | 26.7% (27.3%) | 38.2% (39.1%) | 45.6% |
| feeds with 64 recovered: share of the true 64 found | 71.8% (69.1%) | 50.8% (52.9%) | 50.4% |

The study's reading holds: the method does not misplace events, and a refilled history is not the history a deeper cap would have stored. New: on test only 54% of recovered events are events a deeper cap would have stored, against 73% on the decision sample.

**The brief's "no positional embedding" is wrong, as the study says** [M]: `rec = nn.Embedding(256, d)` is added to every history token (`model.py` l.449, l.550); row norm 2.96 (students) and 2.40 (hard-label), the size of the type and source embeddings; `no_pos` in the checkpoints sets `use_pos_tower = False` only.

## 3. The replay, re-run (item 3)

`s2_passes.py`, `s3_read.py`, `out_s3.txt`. Only the qualifying feeds were scored. All [M], 4-pool unless stated.

**My scores against the study's stored scores** (same candidates, label-free): correlation 1.000000 for a, b, cT, cU, amT, amU on all 8 member x set; largest absolute difference 3.1e-5 on the holdout and 2.7e-4 on val. The study's loader and passes are right.

**The rule:**

| row | n (holdout) | numerator c - b | denominator a - b | rho [my bootstrap 95%] | val numerator | label |
|---|---|---|---|---|---|---|
| **design 1, study events (the verdict row)** | 2,358 | +0.00117 +- 0.00082 (z 1.43) | +0.00262 +- 0.00154 | 0.447 [-0.87, +2.67] | +0.00269 +- 0.00157 | UNDECIDED |
| design 1, my aligner | 2,361 | +0.00120 +- 0.00084 (z 1.43) | +0.00255 +- 0.00154 | 0.472 [-1.07, +2.76] | +0.00300 +- 0.00152 | UNDECIDED |
| design 2, study events | 3,123 | +0.00151 +- 0.00093 (z 1.62) | +0.00238 +- 0.00130 | 0.632 [-0.99, +3.12] | +0.00267 +- 0.00215 | UNDECIDED |
| design 2, my aligner | 3,122 | +0.00164 +- 0.00095 (z 1.73) | +0.00226 +- 0.00131 | 0.724 [-1.02, +4.03] | +0.00322 +- 0.00217 | UNDECIDED |

Holdout and val together: design 1 +0.00173 +- 0.00078 (z 2.23) with the study's events and +0.00187 +- 0.00077 (z 2.41) with mine; design 2 +0.00194 +- 0.00099 (z 1.96) and +0.00223 +- 0.00100 (z 2.22). Numerator positive for 4 of 4 members on both sets in both designs with the study's events (16 of 16), as reported.

**The study's other numbers.** Its stored scores for the passes I did not re-run (d1, d2, dP, g, b192, c192), read with my statistics, return its figures to the last digit (`out_s4.txt`): d1 - a +0.00087 +- 0.00061 and -0.00001 +- 0.00078 pooled; calibration g ratio 1.78 on the holdout and 0.15 on val; fill buckets +0.00098 / +0.00073 / +0.00269 / +0.00585 (design 1) and -0.00021 / -0.00117 / +0.00189 / +0.00436 (design 2). The inference of those passes was not independently re-run.

**Why the rule could not decide** [D]. The length-matched stored reference am - b, which is what perfect recovery of the same number of events would give, reads +0.00158 +- 0.00100 (z 1.58, design 1) and +0.00170 +- 0.00099 (z 1.73, design 2) on the holdout. The OPEN condition "numerator at z >= 2 on the holdout" was out of reach for true events. The study states this; it is a property of the rule and of the sample (3,235 qualifying holdout feeds against the 11,600 the power estimate assumed), not of the study.

## 4. Is the result an artefact of the replay design? (item 4)

Controls declared in block 0, run on the study's events; 4-pool, [M].

| contrast | design 1 holdout | design 1 val | design 1 pooled | design 2 holdout | design 2 val | design 2 pooled |
|---|---|---|---|---|---|---|
| c - b (numerator) | +0.00117 +- 0.00082 | +0.00269 +- 0.00157 | +0.00173 +- 0.00078 | +0.00151 +- 0.00093 | +0.00267 +- 0.00215 | +0.00194 +- 0.00099 |
| bL - b: newest 128, only the length inputs raised to 128 + n | -0.00008 +- 0.00033 | +0.00075 +- 0.00081 | +0.00023 +- 0.00037 | +0.00001 +- 0.00040 | +0.00040 +- 0.00081 | +0.00016 +- 0.00039 |
| cN - b: recovered events without their item embedding | +0.00007 +- 0.00079 | +0.00144 +- 0.00146 | +0.00058 +- 0.00074 | +0.00065 +- 0.00077 | +0.00018 +- 0.00160 | +0.00047 +- 0.00077 |
| c - cN: the item content of the recovered events | +0.00110 +- 0.00089 | +0.00125 +- 0.00186 | +0.00115 +- 0.00089 | +0.00086 +- 0.00105 | +0.00249 +- 0.00214 | +0.00147 +- 0.00104 |
| cX - b: events that restore items of stored 129-256 dropped | +0.00114 +- 0.00078 | +0.00335 +- 0.00162 | +0.00196 +- 0.00077 | +0.00193 +- 0.00089 | +0.00226 +- 0.00200 | +0.00205 +- 0.00093 |
| e - b: another user's events, my donor draw | -0.00048 +- 0.00081 | -0.00168 +- 0.00127 | -0.00092 +- 0.00069 | -0.00014 +- 0.00088 | -0.00186 +- 0.00163 | -0.00079 +- 0.00083 |
| c - e: own minus foreign, my donor draw | +0.00165 +- 0.00095 | +0.00437 +- 0.00182 | +0.00265 +- 0.00090 (z 2.95) | +0.00165 +- 0.00118 | +0.00453 +- 0.00231 | +0.00273 +- 0.00114 (z 2.40) |
| c - am: recovered minus stored, same slots | -0.00041 +- 0.00105 | -0.00032 +- 0.00183 | -0.00038 +- 0.00095 | -0.00020 +- 0.00103 | +0.00002 +- 0.00200 | -0.00012 +- 0.00099 |

- **"Any filler helps": no.** Foreign events read at or below empty slots in both designs with my draw too. The retrieval is not the channel either: with 128 events shown its 16 slots are already filled (`model.py` l.193-195), so pass c does not fill slots that pass b leaves empty.
- **"Length-related inputs differ between passes": not the source.** Raising only log(1 + H) and count / H leaves NDCG where it was.
- **"Pass c restores what pass b removed": no.** 4.8% (holdout) and 6.1% (val) of T events, and 15.5% and 20.1% of U events, are other events on items that occur among the stored events 129-256. Dropping them does not lower the numerator in either design.
- **How much is item content.** Recovered events stripped of their embedding (type, source, category and position kept) give a third of the numerator, inside its noise; the content part alone is 1.3-1.4 SE pooled. Categories of old events are legitimate information, so this is not an artefact; it does mean "the user's own past" is shown by the placebo contrast only.

## 5. Size and parity (item 5)

**Size** [D], train-only pool, effect over the read set x label-free share of feeds in the read set:

| | read-set share, private-like days | effect, holdout + val | pool level, private-like | study |
|---|---|---|---|---|
| design 1 (older than the 256th), study events | 8.15% | +0.00173 +- 0.00078 | +0.00014 +- 0.00006 | +0.00015 +- 0.00007 |
| design 2 (all old-only), study events | 10.87% | +0.00194 +- 0.00099 | +0.00021 +- 0.00011 | +0.00035 +- 0.00015 (bucketed) |
| design 1, my aligner | 8.17% | +0.00187 +- 0.00077 | +0.00015 +- 0.00006 | |
| design 2, my aligner | 10.77% | +0.00223 +- 0.00100 | +0.00024 +- 0.00011 | |

The study's +0.00035 weights the full-fill bucket (+0.00436 +- 0.00196, z 2.2) by its larger share on test (51% of qualifying test feeds against 24% on the holdout). That is a defensible form, but it is the upper end. Range I would quote: +0.00015..+0.0003, only if a retrained model captures the whole worth measured in trained slots. The fine-tuned half of the final file was not replayed by either of us.

**Parity** [M], label-free. The study does not claim OPEN, so its arm sketch is a record, not a specification. Its leak rule is right (train feeds extended from strictly earlier train snapshots only). Its statement that parity cannot be reached is right and is stronger than it reports:

| | train (train snapshots only) | val | test |
|---|---|---|---|
| feeds that would get an extension | 11.58% | 14.98% | 14.35% |
| by day | 1.9% on the first day, rising to 18.8% on day 13 | 14.6% first day, 15.4% second | 15.4 / 14.0 / 13.6 / 14.5% |
| gap to the most recent earlier snapshot, days, 5 / 50 / 95% | 0.10 / 2.82 / 10.41 | 3.05 / 9.60 / 16.53 | 7.08 / 14.16 / 21.28 |
| feeds with an earlier snapshot at least 7 days back | 3.13% (none before train day 7) | 12.77% | 14.19% |

The first val day is not short of earlier snapshots, because train supplies them. The early train days are: a model would see the extension on 2-9% of feeds in the first four train days. The train gap distribution barely overlaps the test one (train 95% quantile 10.4 days, test 25% quantile 10.0). Matching the gap (snapshots at least 7 days older) would leave 3.1% of train feeds, all in train days 7-13. This is the structure of the popularity-staleness lesson (CURRENT_STATUS section 1): an input that is fresher in train than in test.

## 6. Issues

- MAJOR (about the rule, not a fault of the study): the OPEN condition was unreachable on this sample. The stored events in the same slots read z 1.58 and 1.73 on the holdout; the rule's last clause would therefore close the door for lack of power. Whoever records the closure should say so, and not cite the replay as evidence that events beyond the cap are worthless.
- MEDIUM: "every contrast is a lower bound, numerator and denominator alike" is not symmetric. Up to 28 of the 34 precomputed inputs still carry the stored events 129-256 in pass b, and carry nothing about the recovered events in any pass. The denominator and am - b are a marginal worth given those inputs; the numerator is a worth with no such redundancy. If the two paths substitute for each other, rho and c - am are biased upward [I]; "recovered events are worth what the stored events in the same slots are" should read "at most what". Not measurable without rebuilding the inputs.
- MEDIUM: the per-set placebo figures depend on the donor draw. The study quotes design 2 holdout e - b -0.00226 (z -2.60), c - e +0.00376 (z 3.36) and dP - a -0.00252 (z -3.26); my draw gives e - b -0.00014 +- 0.00088 and c - e +0.00165 +- 0.00118 on the same feeds. The pooled own-minus-foreign contrast is stable (z 2.95 and 2.40 against the study's 2.99 and 2.70) and is the only placebo number that should be cited.
- MEDIUM: an implementation component is missing from the SEs. Two aligners that agree on 99% of T events still hand a different recovered list (order or membership) to 20% of holdout and 27% of val read-set feeds, and NDCG on those feeds moves. The 4-pool holdout numerator moves by 0.00003, the val one by 0.0003, family numerators by up to 0.0009 (hard-label, design 1, holdout: +0.00177 with the study's events, +0.00133 with mine, and its "OPEN (sub-row)" label disappears). The study was right not to treat sub-rows as verdicts.
- MEDIUM: the design 2 size (+0.00035) is the bucketed form; the plain form gives +0.00021 +- 0.00011. The headline range should start at +0.00015 and end near +0.0003.
- LOW: one of my controls crosses the OPEN thresholds (design 2 without the item-restoring events, holdout +0.00193 +- 0.00089, z 2.18, rho 0.81, val +0.00226 +- 0.00200). It is a review instrument declared as such, one more row among many near the boundary, and not a verdict; it shows which side of the middle the evidence sits on.
- LOW: the item-content part of the numerator is 1.3-1.4 SE pooled; a third of the numerator survives without item embeddings.
- LOW: on test only 54% of recovered events are events a deeper cap would have stored (73% on the holdout, 60% on val). The study reports the holdout and val figures only.
- LOW: 1-2% of T events on val and test are not older than every stored tail event they can be compared with (1-4% of feeds).
- LOW: the ratio bootstrap is unstable because the denominator is 1.7 SE from zero (mine [-0.87, +2.67], the study's [-1.40, +2.80] for the same row). The rule does not use the interval.
- Not re-derived: the inference of passes d1, d2, dP, g, b192 and c192 (their stored scores reproduce the reported statistics with my code).
- No leak, no rule breach, no undisclosed post-hoc statistic found. The study's three corrections of the brief (positional code, per-type caps, alignment is not a deeper cap) are all correct.

## 7. Recommendation

1. Record UNDECIDED as confirmed. On "one more replay design at most": design 2 was declared before any number and run; this review added an independent aligner, a second donor draw and three controls on the same feeds. The labelled sample on which these members are out of sample (3,235 + 2,160 feeds) is used up, and no further replay design can be decided on it. Treat the allowance as spent [I].
2. No extended-history arm. The prize is +0.00015..+0.0003 at train-only pool level if everything transfers; the train rows cannot be given the test gap; the cost is days of pipeline work and 30-60 GPU-hours. My probability of a pool gain of +0.0003 or more from a 16-student arm: about 0.1 [I] (the study: 0.15).
3. COVERAGE_MAP l.227 should read: "replay undecided for lack of power (holdout z 1.4-1.7, pooled z 2.0-2.4, own-minus-foreign z 2.4-3.0); mechanism probably real; reachable +0.00015..+0.0003; train / test parity not reachable (gap 2.8 days against 14.2)".
4. The 4-snapshot re-cool pilot the study mentions is not free: it needs the model change (positions 257-384), the train and val tables and an identity test before its one GPU-hour. I would not start it; if the lead does, the train extension should be lag-matched (snapshots at least 7 days older) or the result will not speak for test.

## 8. Method notes, hard rules, files

- Declaration: block 0 at 2026-11-15T18:26:36Z and block 1 at 18:32:08Z before the labelled read of 18:37:33Z (`s3_read.py`, run once); block 2 records what was added afterwards. POST-HOC: `s1b_order_parity.py` (label-free), `s4_posthoc.py` (the study's stored scores with my statistics; stability of sub-rows; fill buckets). No decision statistic was added or changed.
- The labelled feeds had been read by the study before me; my passes are further looks at the same labels. Controls bL, cN, cX are review instruments, not a replay design under the rule.
- GPU: one process (PID [n], `/proc` cmdline `.venv/bin/python s2_passes.py --dev cuda`), GPU N only, inference only, 308 s, launched 18:32:12Z under the lane lock with 3 compute processes and 88,734 MiB free; lock released when the process was visible. CPU: one process at a time, 12 threads, a few GB resident (train arrays memory-mapped, only the needed rows read). No process signalled. Writes only in `nn_x/w12/seg/refill_review/` and this report (the lane lock file was opened as the protocol prescribes; no bytecode written anywhere). No network, no install, no submission file, no training.
- Leak rules: only snapshots with a strictly smaller timestamp; user_id 0 excluded; labels read by `s3_read.py` and `s4_posthoc.py` only, on finished score files; members held-out.
- Files: `DECLARATION.txt`, `PROGRESS.md`; `rr_lib.py` (tables, aligner, statistics); `s1_checks.py` -> `out_s1.txt`, `out/myalign_{holdout,val,test}.npz`; `s1b_order_parity.py` -> `out_s1b.txt`; `s2_passes.py` + `run_gpu7.sh` -> `out_s2.txt`, `out_run_gpu7.txt`, `out/rv_scores_<set>_<member>.npz` (16 passes each), `out/rv_meta_<set>.npz`; `s3_read.py` -> `out_s3.txt`; `s4_posthoc.py` -> `out_s4.txt`.
