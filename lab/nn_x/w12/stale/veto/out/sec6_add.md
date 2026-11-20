### 6.1 [R] Resume run: limits, and what is still not checked

- No hz number: the re-cools were running, none had finished when this file was closed.
- CPU stale-val is information-grade by both agents' own rules (mine: one member 0.00008 outside a 0.0003 bar; the witness's b2: 1.04 SE against a 1 SE bar, its B' valid). The two tools give identical files. The scenario leaves `xp_imp_c50` fresh, so every stale-val read of a popularity-light member or admixture is an upper bound; the lagged holdout, which lags that input, reads the W8P20 form at +0.00024 +- 0.00031 on 13,424 feeds [D].
- The row-301 screen is one weight (0.2), one base (W16), train-only level; 36 of the 85 families have 2 or 3 members. Its flags are a chance-level count, not five findings.
- lr: F7 and F9 are different seeds and both pre-date the cooldown mix 0.02; the read is a pool comparison with SE 0.0007.
- The correction of the aux read (section 3.9 c) rests on 7 roots; I did not find why the reference's selected pool is that low beyond the early selected steps, and I did not test other 7-root arms for the same effect (no other arm has 7 roots).
- "8 of 9 arms" in the selection row share four plateaus.
- Still not checked: stale-val of G3 / C3 members and of hard-label 10-epoch members other than base x4; any fine-tuned (FT) branch of a closed arm; PRODDIR / DOMAIN-DIAG, dwa, Ce, proj_mu, C06, C14 and time-gap (taken from the record); the about 300 feature rows of row 323 one by one. `reports/FEATURES_REVIEW.md` was screened for rows where the holdout decided against a positive val read: it_hist_hz and hzcv only (both on the list), plus FR-5, a harness finding and not a closed idea: removing train-period count columns from the GBDT gate base reads val +0.0029 +- 0.0010 and holdout -0.0053 +- 0.0006, the same pattern inside the gate that closed feature rows.
- I did not run `--xt_drop`, the th arm or any GPU job; the two checks of section 5.1 are a code read plus label-free data checks.

### 6.2 [R] Rules on the books that still carry a holdout veto for the staleness family

For the lead; I changed none of these files.
- `docs/05_METHODOLOGY.md` l.54 and l.121 ("Popularity-light pools: decision on val; holdout as a -1 SE veto"), l.119 ("select members and checkpoints on the holdout"), l.125 ("m0, hzcd, a4u, cset, blocks3 net about 0": hzcd is re-opened). Lines 41-51 give every rule family a "holdout >= 0" clause, which is right for arms outside the staleness family.
- `reports/COVERAGE_MAP.md` rows 53, 78, 144, 160, 167, 217, 220, 252 still read CLOSED / REFUTED without a pointer to row 333 (the new policy).
- `nn_x/w12/kd/arm_read8.py` prints a pooled z that includes the in-period holdout; for hz, hz2, kwxd, th the declared rule excludes it: read the columns, not the pooled z.
- `src/pmix.py` (docstring: "report-only + veto ... never promotes"): consistent with the new policy, where pmix is information next to stale-val. The "+1 SE on pmix-private" bar of this audit is the task's bar, not a project rule.

