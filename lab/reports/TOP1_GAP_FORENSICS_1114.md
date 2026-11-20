# TOP-1 GAP FORENSICS (2026-11-14)

Question: is there a data-level signal we are missing that explains the 0.004 public gap to #1 (0.398256 vs our 0.394486)?

**Answer: no.** Nothing in the raw files, the feed structure, labelled-data transfer, the history arrays or the residual slices adds +0.0005 at 2 SE on top of W8. The best of 270+ nested reads is +0.00044 ± 0.00020 (an existing feature column; expected maximum of 248 nulls). The gap is not explained by an unused column, an ordering tell or a leak that we can measure on val.

Scope: CPU only, raw parquet + flat arrays, val labels. Base = W8 = 0.5·z(ft) + 0.5·z(to), val NDCG 0.40165 (reproduced). "Nested" = z(W8) + w·z(s), w chosen on one half of val users and read on the other half, both directions; ± is the paired per-feed SE over 14,423 val feeds. Scripts and logs: `nn_x/w12/forensics/` (p01..p12, `fx.py` harness).

## 1. Verdict table

| # | Probe | Result | Status |
|---|---|---|---|
| 1 | Raw schema vs `data/flat` | all 19 raw columns are flattened; no side table; parquet metadata holds only the Spark schema | null |
| 2 | Raw row order, part files, feed_id | Spearman of row order with feed_id / ts / user / positives: \|ρ\| ≤ 0.013; feed_id is one random numbering over all three splits (max 3,194,169 ≈ total feeds) | null |
| 3 | Stored candidate order in val | positives' relative position 0.5004 ± 0.0019, deciles 0.094-0.110 | null |
| 4 | emb_idx / item_id rank inside the feed | positives' relative rank 0.4996 ± 0.0019 / 0.5011 ± 0.0019 | null |
| 5 | Embedding tables | val and test tables contain exactly the referenced items (0 unreferenced rows of 2.52M / 4.71M); idx order is random vs item_id, first row, first timestamp, candidate count (\|ρ\| ≤ 0.002) | null |
| 6 | `sample_submission.parquet` order | within-feed order identical to the raw test order; feed order is a fresh random permutation (ρ with ts 0.003) | null |
| 7 | Existing feature columns re-read on W8 (248 columns) | best cos_het1 +0.00044 ± 0.00020; z range −3.2..+2.2 | null |
| 8 | Earlier labelled feeds of the same user (train → val) | 5 signals, −0.00036..+0.00010 (table §3) | null |
| 9 | f5 forward echo, generic (leak-class) | +0.00022 ± 0.00027 (sign), +0.00027 ± 0.00031 (sign, < 24 h) | null |
| 10 | f5 floor persistence (leak-class, new mechanism) | real effect, already priced: +0.00001 ± 0.00001 | null |
| 11 | Embedding drift train → val table, directional | 0.00000 (w = 0 chosen in both halves) | null |
| 12 | f1 as a sub-category key for a global item grouping | not a key: 1.02M of 1.11M (feed, category, f1) groups are singletons | refuted |
| 13 | Same-user later-feed leak on test | flagged events look exactly like val's; organisers' rule appears label-aware (§4) | not exploitable |

## 2. Where W8 loses NDCG (val, loss = 1 − NDCG, total 0.5983)

Loss is spread in proportion to feed share; no slice is both large and anomalous.

| Slice | Feeds | NDCG | Share of loss |
|---|---:|---:|---:|
| M 50-99 | 22.1% | 0.365 | 23.4% |
| M 100-199 | 20.9% | 0.308 | 24.2% |
| M 200+ | 16.2% | 0.278 | 19.5% |
| M < 30 | 26.9% | 0.47-0.69 | 19.5% |
| Positive has zero leave-user-out popularity | 25.3% | 0.367 (vs 0.413) | 26.8% |
| Positive not a train candidate | 25.2% | 0.393 (vs 0.405) | 25.5% |
| Positive's category < 10% of the feed | 17.5% | 0.26-0.33 | 20.3% |
| Positive older than feed median (f4 rank-z < −0.3) | 31.2% | 0.368 (vs 0.417) | 32.9% |
| User seen in train | 30.4% | 0.393 | 30.8% |
| Guest (user 0) | 4.6% | 0.408 | 4.5% |
| Best positive ranked 8+ by W8 | 50.8% | 0.16-0.27 | 65.6% |

- Users seen in train are slightly *worse* after adjusting for M and positives: 1 train feed −0.005 ± 0.004, 2-3 feeds −0.015 ± 0.006. Their labelled past does not help (§3).
- Guests are better than expected (+0.023 ± 0.009 adjusted), so the short-history bucket is not a loss centre.
- Day and hour are flat (Tue 0.402, Thu 0.401; 4-hour bins 0.387-0.416, SE 0.004-0.008).
- Group shares on test match val: seen-in-train 0.303 public / 0.288 private; guests 0.041 / 0.041.

No simple cue beats the model inside a slice: the 248-column sweep is the per-cue test, and its distribution is a null.

## 3. Labelled-data transfer (train → val, same user)

Coverage: 30.4% of val feeds have an earlier labelled feed (public 30.4%, private 28.8%); 8.4% of val feeds contain a re-shown (user, item) pair (public 7.8%, private 6.8%); 0.18% of val rows are re-shown.

| Signal | Nested delta |
|---|---|
| item shown to this user before | +0.00007 ± 0.00006 |
| shown before, weighted by best production position | +0.00009 ± 0.00006 |
| user × category click rate from earlier feeds (shrunk) | −0.00036 ± 0.00019 |
| user × category earlier clicks | −0.00030 ± 0.00018 |
| user × category earlier exposure | +0.00010 ± 0.00009 |

Re-shown rows click at 1.64% vs 1.23% for other rows of the same feeds, but W8 already scores them at z +0.49. Only 16 val rows are re-shows of an earlier click.

## 4. Leak-class channels

**Same-user later feed (known, re-examined).** 103 test feeds are flagged (57 public, 46 private). Their flagged history events have the same signature as val's: 178 of 235 are type 1 / source 1, as are 33 of 42 in val, where 41 of 42 are positives. So the test histories were not altered. The overlay submission did re-rank 54 public feeds, yet scored 0.392819 vs 0.392820.

Inference (not verified): a −0.000001 change is a total of about −0.009 NDCG over 8,562 feeds. That fits a label-aware rule: feeds (or positives) whose click is visible in a later test history are removed from scoring, and the few flagged feeds whose flagged items are negatives stay and lose slightly. A blanket "drop users with two feeds" rule would give exactly zero. Either way there is nothing left to exploit, and the organisers evidently audited this channel.

**f5 forward echo, new mechanism.** f5 has two atoms: a floor (0.0161, 10.8% of val rows, target rate 0.9%) and a cap (0.9839). It updates in batches: 53% of same-item pairs less than 6 h apart have identical f5.
- Floor now and still floor in a later feed: target rate 0.157% (n 10,212) vs 1.34% when a later feed shows it above the floor.
- This is a true "not clicked yet" certificate, but W8 already scores these rows at z −1.27. Hard demotion gives +0.00001 ± 0.00001.
- Using test feeds as the future of val gives +0.00048 ± 0.00022. That read is val-only: test has no later split. The within-split analogue is the 0.00001 above.
- Test coverage (still-floor rows): public 1.10%, private 0.93%, val 0.74%. Rows with a later f5 increase: public 9.3%, private 4.4%, val 3.9%. Public has more coverage than private, so any team using this family is flattered on public, but the measured value is nil.

**Embedding drift.** 65.4% of val candidates have a train-table embedding; 77.1% of those are bit-identical. For drifted rows the target rate is flat across quintiles of cos(u, e_val) − cos(u, e_train): 1.71-1.91%.

## 5. History arrays

The raw history has no time field; `hist_hpos` is the array index built in `src/prep.py`. Order, type, source and category are all model inputs. Nothing is dropped between raw and flat. Snapshot alignment, time gaps and session bounds were closed on 11-13 and were not redone.

## 6. What this means for the gap

- A data channel is now unlikely as the explanation. Remaining candidates are a better model or training protocol, or public noise on a single entry.
- Rough noise framing (not measured here): cross-team paired public SE was earlier put at 0.0012-0.0018, so a 0.0031 lead over #2 is about 2 SE; some of it should regress on private.
- One small oddity, no ranking value: row order inside raw parts correlates with M (ρ 0.026 ± 0.008 val, 0.032 ± 0.006 test). It is a feed-level property and cannot change a within-feed order.

## 7. Not done

- Guest / cross-user-id stitching by history overlap was not redone (closed in `eda_history_leak.md`: 6 + 17 pairs).
- No test of feed-level signals interacting with candidates beyond the 248 existing columns.
- The neutralisation rule in §4 is an inference from one pair of public scores.
