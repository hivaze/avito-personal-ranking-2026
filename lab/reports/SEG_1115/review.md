# SEG_1115 / review: adversarial review of the user-information segment study (`diag.md`)

Written 2026-11-15T17:40Z. Reviewer directory `nn_x/w12/seg/review/` (scripts, raw outputs `out_*.txt`, `DECLARATION.txt` with eight stamped blocks, `PROGRESS.md`). CPU only. Everything below was computed with my own code (`rv_lib.py`: own per-feed z, own NDCG checked against `src/metrics.py`, own cluster SE by user, own AUC, own cosine statistics); the study's `lib.py` was not imported. Marks: [M] measured here, [D] arithmetic from [M], [I] judgement. SE are cluster-robust by user unless stated. "Cap" = H = 256 (28.95% of val, 28.81% of test), NEAR = H 100-255, REST = H < 256.

## 0. Verdict

**confirmed = false**, by the rule I wrote before computing anything (DECLARATION block 0): one statement of the study fails a direct test. Most of the study stands.

1. **Stands, reproduced to the fourth decimal** [M]: the segment table; FINAL 0.40271 from its 20 member files; cold, anonymous and sparse users are small and not weak (H = 0: 0.86% of test, 0.3784 +- 0.0176; H 1-9: 0.4431 +- 0.0173; user_id 0: 4.12% of test, 0.4093 +- 0.0101); no stored family beats FINAL on a small segment when chosen nested (my nested read -0.0073 / -0.0027 / -0.0030); the weak cold cell (H = 0, all-zero user embedding) is 0.20% of test.
2. **Stands, with better evidence than the study gave** [M]: the gap at the cap is the same in every model family. Paired contrast against FINAL, cap minus NEAR: held-out twin W16P20w +0.0008 +- 0.0009, hard-label teacher +0.0019 +- 0.0014, G3 file +0.0014 +- 0.0013. Nothing in our recipe (distillation, popularity-light admixture, full data) moves it. NDCG falls steadily with H among users whose history is NOT truncated (0.4139 / 0.4067 / 0.3989 for H 100-149 / 150-199 / 200-255, cap 0.3906), so most of the "cap gap" is the continuation of a trend in the kind of user. The lead's +0.005 is not available.
3. **Does not stand: statement 7, "no sign of history starvation at the cap".** A direct model test says the opposite [M]. Removing the oldest 64 of the 256 stored events at inference costs the capped feeds -0.0023 +- 0.0008 NDCG (six members, z -2.9; 6 of 6 negative; FINAL's own members -0.0020 +- 0.0010; hard-label members -0.0025 +- 0.0009). The loss is information, not a distorted length signal: with the history length kept and only the embeddings of those events removed it is -0.0028 +- 0.0008 (z -3.5). The marginal 64-event stretch is worth as much at depth 193-256 (-0.0023) as at depth 129-192 (-0.0025): no decay at the boundary of what the data stores. The study's three supports were proxies: the cosine ranker loses nothing from 256 to 64 events where the model loses -0.0090 +- 0.0014.
4. **The closure the study leans on is not evidence.** "Longer history from earlier snapshots" (COVERAGE l.227, err_segments section 6) was closed with a probe of the form `score + w z(similarity to the old-only items)`. Calibrated on an effect that is known to exist (the model's own use of events 193-256), that probe returns -23% .. +36% of the truth and is never 2 SE from zero [M]. An NN that reads the old events itself was never tested.
5. **Size of what is reachable** [D]: capped feeds whose user has a strictly earlier snapshot are 14.3% of test (private 14.0%); on val 71% of such feeds have at least 64 old-only items. If the events just beyond the cap are worth what events 193-256 are (not verified), a model trained with them gains about +0.0002..+0.0005 at pool level; if not, 0. Val cannot resolve that at pool level (8-pool paired SE 0.00047).
6. **Audit: "0 of 90, inside noise" stands and now has an independent sample behind it** [M]. On the 30,000 holdout feeds (cap bucket fixed in advance, SE half of val's) the two val cells that carried the positive tail vanish: unique retrieval slots (a4u, 10 epochs) val +0.00274 -> holdout -0.00060 +- 0.00057; GRU under distillation val +0.00216 -> holdout +0.00000 +- 0.00050. The study's "replication F1" of a4u was not independent (the re-cools start from the same snapshots).
7. **Proposals.** kwa4u: do not run (my prior 0.02 [I]; its kill rule has power 0.13-0.24 at the effect its own prior names). kwr32: do not run (retrieval depth 32 reads -0.00049 +- 0.00076 at the cap on the holdout). No GPU arm follows from this review either.
8. **One CPU follow-up is the only open door**: a zero-training "refill replay" (section 6) that decides whether events older than the stored 256 carry model-usable value. Nothing else for sparse or heavy users survived sizing: "borrowed history" for returning sparse users covers 0.05% of test feeds; "functionally cold" feeds are 0.16% of val.

## 1. Reproduction (task item 1)

`out_r0.txt`, `out_r1.txt`, `out_b.txt`. R0 PASS: every cell of the study's segment table is reproduced exactly (own NDCG equals `src/metrics.py` to 0.0 per feed; rebuilt score differs from the stored branches by 6e-7).

| number of the study | study | mine [M] | verdict by my declared rule |
|---|---|---|---|
| FINAL val | 0.40271 | 0.40271 | PASS |
| D(cap - REST) / D(cap - NEAR) | -0.01711 +- 0.00443 / -0.01549 +- 0.00470 | identical | PASS |
| skill ratio cap / NEAR: FINAL, pre2_fe, max cosine | 0.872 / 0.872 / 0.859 | 0.872 (0.035) / 0.872 (0.048) / 0.859 (0.062); R_FINAL - R_X z 0.00 / +0.21 | CONFIRMED |
| A1 residual, label-free set + activity markers, vs NEAR | -0.0063 +- 0.0042 (41% left) | odds weighting -0.0074 +- 0.0058 (48%); regression form -0.0077 +- 0.0042 (50%) | CONFIRMED (inside the declared band) |
| A1 residual without activity markers, vs NEAR | -0.0093 +- 0.0042 (60%) | -0.0108 +- 0.0055 (69%); regression -0.0100 +- 0.0042 (65%) | same reading |
| cosine ranker, newest 64 minus all 256, cap | +0.0008 +- 0.0021 | +0.0008 +- 0.0021 | CONFIRMED as a number |
| mean top-32 minus top-16 cosine, AUC, cap | -0.0045 +- 0.0012 | -0.0045 +- 0.0012 | CONFIRMED as a number |
| 20% history drop, cap minus NEAR | -0.00001 +- 0.0009 | -0.00001 +- 0.00091 | CONFIRMED as a number |
| part B: pop-light pool - FINAL at H = 0; FD16 - FINAL at H 1-9 | -0.0055 +- 0.0082; -0.0054 +- 0.0021 | identical | PASS |

My A1 estimator is independent of the study's (propensity of cap vs reference by LightGBM, 5 folds by user, odds weights clipped at p99, effective sample 2,881 of 7,165 NEAR feeds; covariates rebuilt, all history covariates on the newest 64 events only). After weighting every matched covariate is balanced (|SMD| <= 0.11).

## 2. Is "intrinsic, not a deficit" supported? (task item 2)

Same reads on other files (`out_r1.txt`) [M]:

| file | NDCG all | D(cap - NEAR) | skill ratio (boot SE) | paired contrast vs FINAL, cap minus NEAR | A1 residual (my XB, vs NEAR) |
|---|---|---|---|---|---|
| FINAL (FD16P20) | 0.40271 | -0.0155 +- 0.0047 | 0.872 (0.035) | | -0.0074 +- 0.0058 |
| W16P20w (held-out twin, rebuilt as build_r4.py) | 0.40214 | -0.0147 +- 0.0047 | 0.877 (0.035) | +0.0008 +- 0.0009 (z +0.83) | -0.0069 +- 0.0058 |
| W16 (no popularity-light) | 0.40158 | -0.0150 +- 0.0047 | 0.874 | +0.0005 +- 0.0010 | |
| teacher, 11 hard-label members | 0.40054 | -0.0136 +- 0.0047 | 0.885 (0.035) | +0.0019 +- 0.0014 (z +1.37) | -0.0061 +- 0.0058 |
| G3 file (hard labels, FT + TO) | 0.40032 | -0.0141 +- 0.0047 | 0.881 | +0.0014 +- 0.0013 (z +1.05) | -0.0060 +- 0.0058 |
| one hard-label member | 0.39632 | -0.0144 +- 0.0047 | 0.873 | +0.0011 +- 0.0021 | |

SAME ANSWER by my declared rule (all paired |z| < 2, ratios within 0.03, residuals inside FINAL's band). The paired contrast is the sharp form of the question and it is tight: the cap gap does not depend on which of our model families ranks.

Three qualifications the study should carry:

- **The "shared difficulty" label depends on the scale.** On the ratio scale FINAL equals the simple rankers. In absolute NDCG FINAL loses more at the cap than every simple ranker: paired contrast (simple ranker minus FINAL, cap minus NEAR) pre2_fe +0.0064 +- 0.0040, max cosine +0.0088 +- 0.0043 (z 2.1), popularity +0.0113 +- 0.0048 (z 2.4), exposure +0.0117 +- 0.0044 (z 2.6) [M]. Neither scale is privileged, and the ratio rule was written after the ratios were seen (the study says so). The robust fact is the model-family invariance above, not the ratio.
- **NEAR is not a homogeneous reference** (`out_s.txt`) [M]: H 100-149 0.4139 +- 0.0054, 150-199 0.4067 +- 0.0052, 200-255 0.3989 +- 0.0047, cap 0.3906 +- 0.0037. Cap minus H 200-255 is -0.0083 +- 0.0060 (1.4 SE); H 200-255 minus H 100-149 is -0.0151 +- 0.0072. Neither of my two declared readings ("step" or "within 1 SE") is met; the profile is a steady decline. The share of users seen in train rises along it (19 / 28 / 36 / 52%). This is the cleanest support for "kind of user" and the study does not show it.
- **Covariate with no common support.** `s1_feats.py` computes "share of candidates whose category occurs in the history" on the whole history (`hc = hcat[ho:ho + h]`), against the study's own declaration (newest 64 events). With newest-64-only covariates the residual is a little larger (48-50% left instead of 41%). "40-63% of the gap goes with the kind of user" should read "about half; the other half is 1.3-1.8 SE from zero".

## 3. The statement that fails: history starvation at the cap

### 3.1 Truncation dose-response with the model itself (T)

`trunc_replay.py`, `t_read.py`, `t_fd_read.py`. CPU fp32 inference of stored checkpoints through `nn_x/w5/{model,data}.py` (read-only), history cut to the newest K events with the loader's own `max_hist` (masks, H and every in-model history input follow). Reference passes correlate 1.000000 / 0.999999 with the stored train-only val files. The precomputed xt inputs keep the full history, so every cost below is a lower bound in magnitude. Rule written before the run: "the cap binds for the model" if the cost of K = 192 at the cap is <= -0.0010 with z <= -2. My written expectation was "does not bind"; it was wrong.

| members | K = 192, cap | K = 128, cap | K = 64, cap | K = 192, H 193-255 |
|---|---|---|---|---|
| held-out distilled students (kd05w192 s951, s952) | -0.00234 +- 0.00092 (z -2.5) | -0.00437 +- 0.00126 | -0.00930 +- 0.00166 | -0.00077 +- 0.00066 |
| hard-label members (w5_e10_base s951, s952) | -0.00246 +- 0.00093 (z -2.7) | -0.00475 +- 0.00122 | -0.00769 +- 0.00158 | -0.00163 +- 0.00068 |
| FINAL's own members (w5_fd10_kd05w192 s951, s952) | -0.00202 +- 0.00098 (z -2.1) | -0.00532 +- 0.00131 | -0.00985 +- 0.00163 | -0.00016 +- 0.00065 |
| all six | -0.00227 +- 0.00077 (z -2.9) | -0.00481 +- 0.00109 | -0.00895 +- 0.00144 | -0.00085 +- 0.00050 |

All [M]. Per member at K = 192: -0.0010, -0.0031, -0.0013, -0.0034, -0.0015, -0.0035 (6 of 6 negative). Marginal worth of each 64-event stretch at the cap, six members: events 193-256 -0.0023 +- 0.0008, events 129-192 -0.0025 +- 0.0008, events 65-128 -0.0041 +- 0.0010. Over all val feeds the oldest quarter is worth -0.00083 +- 0.00025 (z -3.4) at pool level.

### 3.2 Information or a distorted length signal? (TN)

Truncation also changes log(1 + H) and the count inputs. Control declared before it was run (block 2): keep the stored length, type, source and category of every event, remove only the embeddings of the events at index >= K (the loader's representation of an event without an embedding). `trunc_null.py`, `tn_read.py` [M]:

| members | embeddings removed at index >= 192, cap | at index >= 128, cap | truncation minus this (length-signal part), K = 192 |
|---|---|---|---|
| held-out distilled students | -0.00219 +- 0.00094 (z -2.3) | -0.00374 +- 0.00131 | -0.00015 +- 0.00062 |
| hard-label members | -0.00334 +- 0.00094 (z -3.6) | -0.00509 +- 0.00125 | +0.00088 +- 0.00066 |
| FINAL's own members | -0.00278 +- 0.00097 (z -2.9) | -0.00536 +- 0.00132 | |
| all six | -0.00277 +- 0.00079 (z -3.5) | -0.00473 +- 0.00112 | |

INFORMATION by the declared rule. The cost of truncation is the content of the old events.

### 3.3 Why the study read the opposite

- "Cosine ranker on the newest 64 = all 256 (+0.0008 +- 0.0021)": true for a max-cosine ranker. The model loses -0.0090 +- 0.0014 for the same cut. A proxy null was used to dismiss a model-level effect; this is the mirror image of the TRUNC lesson (a proxy gain does not justify GPU; a proxy null does not close a mechanism).
- "Dropping 20% of events costs the same at the cap (-0.0020) as below it (-0.0020), so the model is not starved": the number is right, the inference is not. Equal marginal value means every stored event has value; the capped users are the only ones from whom the data withholds more. (The 10% pass, declared with the 20% one and not quoted in the summary, reads -0.00121 at the cap against -0.00069 at NEAR, contrast -0.00052 +- 0.00078.)
- "Events ranked 17-32 by cosine add nothing": this part holds, and the holdout agrees (section 4): retrieval depth does not bind. History length does.

### 3.4 What the probe behind the l.227 closure can see (PC)

`pc_probe.py`. On the cap feeds take S_192 (members deprived of events 193-256) and try to recover the known truth G_true = NDCG(S_256) - NDCG(S_192) with `S_192 + w z(q)`, w nested by user, exactly the form that closed the earlier-snapshot extension [M]:

| q | distilled students, G_true +0.00250 +- 0.00109 | hard-label members, G_true +0.00147 +- 0.00110 |
|---|---|---|
| max cosine to events 193-256 | -0.00035 +- 0.00052 (-14%) | -0.00030 +- 0.00047 (-20%) |
| mean cosine to them | +0.00091 +- 0.00068 (36%) | -0.00041 +- 0.00058 (-28%) |
| share of them in the candidate's category | 0 (w = 0 picked) | 0 |
| gain in max cosine over the newest 192 | -0.00057 +- 0.00044 (-23%) | +0.00030 +- 0.00039 (20%) |
| best (q, w) nested | +0.00091 +- 0.00068 (36%) | -0.00034 +- 0.00056 |

PARTLY (distilled) and BLIND (hard labels) by the declared rule. The closure numbers of l.227 (-0.0001 / -0.0003 / -0.0002, SE 0.0002-0.0003) are what this probe returns for an effect of +0.0015..+0.0025 that is there. The sentence in `diag.md` section 5, "the earlier-snapshot extension is closed and the model's marginal use of events is lower at the cap, not higher", is wrong on both counts.

### 3.5 Size [D]

`out_z.txt`. Capped AND user has a strictly earlier snapshot: 14.96% of val, 14.31% of test (public 15.03%, private 14.00%). For the val ones the earlier train snapshots add a median of 106 old-only distinct items; 71% have >= 64, 40% have >= 128 [M]. If events 257-320 are worth what events 193-256 are (0.0023-0.0028 per capped feed) and a retrained model uses them equally well: 0.143 x 0.71 x 0.0025 = +0.00025, plus the second stretch for the 40% = about +0.0004 at pool level; rounded range +0.0002..+0.0005 [D], 0 if old-only events are worthless. This replaces the lead's +0.005. It is unverified in the one step that matters (events beyond the stored 256), which only earlier snapshots can test.

## 4. By-bucket reads of stored arms (task item 3)

`r3_holdout.py`, `out_r3_holdout.txt`.

- My re-computation of six val cells equals the study's to five decimals. The 90-cell Bonferroni reading is sound.
- **F1 is not an independent replication.** `nn_x/r6/logs/r6_cd_m02_a4u_r101.json` has `"init": ".../r6_a4u_e10_s101@77500.pt"`: the m02 re-cools are the same six runs as the 10-epoch cell with another cooldown. The study calls it "independent replication".
- **A structural hint the study's reading missed.** Grouped by control pool, the history-path arms have a positive cap-minus-rest interaction in 6 of 6 groups (per-control mean z +1.43, +0.20, +1.25, +0.30, +1.00, +1.65; combined 2.38 against my bar of 2.5). Close enough that I added an independent test, declared before any holdout number.
- **Holdout replication (HR).** 30,000 in-period holdout feeds, 11,600 at the cap, other users than val (0.2% overlap), bucket fixed in advance; these are architecture contrasts, so the holdout is an admissible witness [M]:

| cell | val, cap | holdout, cap | holdout, cap minus rest | rule |
|---|---|---|---|---|
| GRU history encoder under distillation (4 seeds) | +0.00216 +- 0.00109 | +0.00000 +- 0.00050 | +0.00029 (z +0.4) | REFUTED |
| a4u, 10 epochs (6 seeds) | +0.00274 +- 0.00119 | -0.00060 +- 0.00057 | -0.00019 (z -0.3) | REFUTED |
| a4u m02 re-cools (same snapshots) | +0.00115 +- 0.00121 | +0.00014 +- 0.00056 | +0.00078 (z +1.1) | UNDECIDED |
| kv64, 2 epochs | +0.00201 +- 0.00101 | +0.00084 +- 0.00047 | +0.00017 (z +0.3) | replicated as a cell, not a cap effect |
| retrieval depth 32, 2 epochs | +0.00163 +- 0.00105 | +0.00049 +- 0.00051 | +0.00060 (z +0.9) | UNDECIDED |
| retrieval depth 32 / 32u, 10 epochs | +0.00052 / +0.00075 | -0.00049 +- 0.00076 / -0.00001 +- 0.00078 | | REFUTED |
| 2 history layers under distillation | -0.00013 +- 0.00127 | -0.00135 +- 0.00059 (z -2.3) | | REFUTED |
| history-drop 0.3 under distillation | -0.00178 +- 0.00110 | -0.00048 +- 0.00049 | | (drop arm, information) |

  Family statistic +0.21 (bar 2): NOT SUPPORTED. No history-capacity arm gains at the cap on an independent sample. "0 of 90" stands.
- The cap bucket itself was chosen from the lead's table on the same val feeds. The arm reads are differences against controls, so the conditioning is mild; the holdout read removes it.
- A by-product, checked against a closed family and closed again: the fine-tuned branch alone is below the train-only branch at the cap in every base member set (FD16 -0.0027 +- 0.0011, W16 -0.0029 +- 0.0011, G3 -0.0020 +- 0.0010; one observation of about z 2.4, since the sets share feeds and folds). At blend level it does not carry: w_ft 0.3 minus 0.5 at the cap +0.00057 +- 0.00047 (FD16P20) and +0.00025 +- 0.00043 (twin). The MIX_1115 closure of w_ft-by-H stands.

## 5. The proposed arms (task item 4)

A/A null from random splits of the 16 held-out distilled students (`r4_aa.py`) [M]: paired cluster SE 4 vs 4 pools 0.00056 overall, 0.00106 on the cap (the lead's 0.0006 / 0.0011 confirmed); 8 vs 8: 0.00047 / 0.00087. Power of "z >= 2" with seed noise and feed sampling: +0.0005 overall 0.24 (8-pool), +0.0010 overall 0.55, +0.0022 on the cap 0.66.

| arm | code path | cost | kill rule decidable? | my prior of a pool gain >= +0.0005 [I] |
|---|---|---|---|---|
| kwa4u (`--retr_uniq --retr_tcnt` under distillation) | exists: `train.py` l.186-188 and `train_full2.py`, `model.py` HistRetrieval / history_dedup; the assert at l.322 only forbids the flags with `--init`; fine-tune path works (stored a4u OOF files). No `E:kwa4u` line in `lane.sh` yet. The study's description is right | stored kd05w192 logs: 77-245 min per student, so 8 students = 10-33 GPU-hours (the study's 10-28 is right) | NO at +0.0005 (power 0.13-0.24). The cap secondary (+0.0022) has power 0.66 but the cell it comes from is refuted on the holdout | **0.02** (study: 0.05). Do not run |
| kwr32 (`--retr_m 32`) | `lane.sh` l.135 | same | no | 0.02; refuted at the cap on the holdout. Do not run |
| kwh6 (`--heads 6`, d192) | `lane.sh` l.136 | same | no | 0.03; not a segment arm, nothing here speaks to it |
| kw256d2 (d256, dropout 0.2) | `lane.sh` l.134; parked x4 | wider: longer | no | 0.03-0.05; not a segment arm; leave to the capacity queue's own rule |

## 6. Missed by the study and by the coverage map (task item 5)

1. **Heavy users: the cap binds for the model and the lifelong-history closure rests on a blind probe** (section 3). Not in COVERAGE_MAP as an open mechanism; l.227 lists it as closed. The decisive test needs no training:
   **Refill replay (CPU).** For capped feeds whose user has a strictly earlier snapshot, find the events immediately older than the stored 256th by aligning the stored history with the earlier snapshot's history (item id + event type sequences; snapshots with a strictly earlier timestamp only). Score with held-out members only (2 distilled + 2 hard-label; never full-data members on holdout feeds), three passes: (a) the stored 256 events; (b) the newest 128 only; (c) the newest 128 plus, in slots 129-256, the 128 events from beyond the cap (the model was trained with at most 256 positions, so the old events have to be shown inside them). Read rho = [NDCG(c) - NDCG(b)] / [NDCG(a) - NDCG(b)]: the share of the worth of the true events 129-256 that the events beyond the cap reproduce. The same with 192 + 64 as a second line. Sample: the 30,000 holdout feeds (11,600 capped; needs a subset loader, the whole train split with xt does not fit in 30 GB on CPU) plus the 2,158 capped-and-seen val feeds. Val alone is under-powered (full recovery at K = 128 on 860 feeds with >= 128 old-only items is about 1.5 SE); on the holdout full recovery is about 6 SE, half recovery about 3 SE [D]. Rule to write before the run: OPEN an extended-history arm if rho >= 0.5 with the numerator at z >= 2 on the holdout and the same sign on val; CLOSE for good with a model-based reason if the numerator's upper 95% bound is below 25% of the denominator. Cost: about half a day of one CPU process. Only if OPEN: students with max_hist 384 on stored history + old-only events, train rows built by the same rule (parity risk: train feeds have fewer earlier snapshots than test), read on the capped-and-seen bucket (15% of val, 8-pool SE about 0.0012, expected +0.0023..+0.0045 there). My prior for a pool gain >= +0.0003: 0.15; >= +0.0005: 0.05 [I].
2. **Sparse users: borrowed history.** The stored history behaves like a time window capped at 256, so a returning user can show H = 0 now while an earlier snapshot holds a history (24.7% of val H = 0 known users have one, median H 256). Sized label-free before any run (`out_z3.txt`): known users with H <= 19 and >= 10 old-only items are 0.173% of val and **0.049% of test** feeds. Below my 0.5% gate: recorded, not worth a run.
3. **"Functionally cold" feeds** (H >= 10 but at most 10% of candidates matched by any history event at cosine 0.8): 0.16% of val. H captures low user information; there is no hidden segment.
4. **Training mix.** 42.3% of train feeds are at the cap against 28.8% of test; H 100-255 is 37.8% against 50.5% [M]. The model is trained more, not less, on capped users, so under-exposure does not explain the gap. Activity-parity weighting of the training sample is not in the coverage map (nearest: DW3 domain re-weighting, closed); nothing here supports it, prior 0.03 [I].
5. An earlier report already had the cap finding and is not cited: `reports/err_segments.md` ("H = 256 feeds (28.7%), M-adjusted -0.0145 (0.0030)").

## 7. Corrected answer to the owner's question

Cold-start and unknown users are handled by design (no user-ID embedding, a user_id-0 flag, log H, the organisers' user embedding, item-side inputs) and they are neither large nor weak: together under 5% of test, with a ceiling of about +0.0002 even under generous assumptions. This part of the study is correct and was researched adequately. For heavy users the study's conclusion "nothing to gain, the difficulty is intrinsic" is half right: most of the gap is the kind of user and no architecture change helps (checked on an independent sample), but the model does lose measurable accuracy because the data stores only 256 events, and the one way to give it more (earlier snapshots of the same user, for about 14% of test feeds) was closed by a probe that cannot see such an effect. Reachable gain if it works: +0.0002..+0.0005 [D]; decided by one CPU test.

## 8. Issues

- MAJOR: statement 7 ("no sign of history starvation at the cap") fails a direct model test: -0.0023 +- 0.0008 for the oldest 64 stored events (z -2.9, 6 of 6 members, FINAL's own members included), information by the embedding-only control (-0.0028 +- 0.0008), no decay of the marginal stretch between 129 and 256. The study's supports were cosine proxies and a misread of the history-drop equality.
- MAJOR: the study relies on the l.227 closure of the earlier-snapshot extension; the probe that produced it recovers -23% .. +36% of a known effect of the same kind and is never 2 SE from zero. `diag.md` section 5 states the model's marginal use of events is "lower at the cap, not higher"; per removed stretch it is equal or higher.
- MAJOR: the optional arm kwa4u is motivated by a cell that an independent sample refutes (holdout cap -0.00060 +- 0.00057 against val +0.00274), its "replication F1" reuses the same snapshots, and its kill rule cannot decide the effect it names (power 0.13-0.24). Prior 0.02, not 0.05.
- MEDIUM: "shared difficulty" holds on the ratio scale only; in absolute NDCG FINAL loses 0.006-0.012 more at the cap than every simple ranker (z 1.6-2.6). The ratio rule is post hoc (disclosed by the study). The evidence that carries the conclusion is the model-family invariance, which the study did not test on the twin.
- MEDIUM: NEAR is treated as a homogeneous reference; NDCG falls by 0.015 inside it. The study's -0.0155 overstates what is specific to the cap (cap minus H 200-255: -0.0083 +- 0.0060).
- MEDIUM: A1 covariate "candidate category in history" is computed on the whole history (no common support across H), against the study's declaration. Corrected residual: about half the gap left, not 41%.
- MEDIUM: the 30,000-feed holdout, admissible for architecture contrasts and twice as precise at the cap, was not used for the audit.
- LOW: the 10% history-drop contrast (-0.00052 +- 0.00078, cost 1.75x at the cap) was declared with the 20% one and left out of the summary.
- LOW: `reports/err_segments.md` (same finding, 11-13) is not cited; the heavy-user gap reads as new.
- LOW: the generous ceiling for small segments is right; "nested gain + 2 SE" bounds only a switch among stored families, not a dedicated treatment. Immaterial at these shares.
- No leak found; no rule breach found; the study's own disclosures of post-hoc rules are accurate.

## 9. Recommendation, in order

1. GPU: nothing from this study. Do not run kwa4u or kwr32. kwh6 / kw256d2 are capacity arms with no segment evidence; leave them to their queue.
2. Record: mark COVERAGE l.227 as "closed on a probe that recovers at most 36% of a known effect; NN form untested; reachable +0.0002..+0.0005 [D]", and add the rows of section 6 (borrowed history sized at 0.05% of test; functionally cold 0.16%; history-capacity arms refuted at the cap on the holdout).
3. CPU, optional, the only open door for heavy users: the refill replay of section 6 with its rule written first. Open a GPU arm only on rho >= 0.5.
4. Stop treating the heavy-user gap as a +0.005 target; the trend in H among uncapped users says it is mostly the kind of user.

## 10. Method notes, deviations, files

- Declaration: block 0 (17:05:14Z) before any number of mine; blocks 1-7 between runs. Disclosed deviations (block 7): the grouping in my audit meta-read was ambiguous as declared (I used per-control means and added the holdout test because the statistic was close); the cross-member-set replication of the fine-tune contrast was computed without its own rule (the decision-bearing blend read was declared first); the splits of the truncation cost by "seen in train" are descriptive; the two confounds of the truncation read were not anticipated in block 0 and their controls were declared after it was read, before they were run.
- Limits of the truncation read: inference-time intervention on models trained with full histories; xt inputs unchanged (lower bound); val only (the holdout form needs a subset loader). It shows that stored old events carry undiminished value, not that events beyond the stored 256 do.
- Hard rules: CPU only (`CUDA_VISIBLE_DEVICES=""`), at most one heavy process at a time plus one light one, 12 threads, memory far below 30 GB, no process signalled, nothing written outside `nn_x/w12/seg/review/` and this report, no network, no install, no submission file. Val labels used descriptively or inside 5-fold user-grouped procedures; no later snapshots of any user.
- Files: `rv_lib.py`, `r0_repro.py`, `r1_cos.py`, `r1_reads.py`, `r3_holdout.py`, `r4_aa.py`, `trunc_replay.py`, `trunc_null.py`, `t_read.py`, `tn_read.py`, `t_fd_read.py`, `pc_probe.py`, `s_missed.py`, `z_sizing.py`, `z2_rep.py`, `z2b.py`, `z3_sparse.py`, `b_check.py`; outputs `out_*.txt`; arrays in `out/` (truncated-history scores in `out/trunc/`).
