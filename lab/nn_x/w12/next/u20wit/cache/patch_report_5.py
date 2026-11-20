p = '<repo>/reports/NEXT_1115/u20wit.md'
s = open(p).read()
R = [
 ("| short, re-weighted to the val M < 50 size mix | eff. 9,366 | | | +0.00011 ± 0.00094 | | +0.00000 ± 0.00095 | | |",
  "| short, re-weighted to the val M < 50 size mix | eff. 9,366 | | | +0.00011 ± 0.00094 | | +0.00000 ± 0.00095 | | |\n| short, scored with the val / test domain token ('rl'; `out/rl_short_read.txt`) | 11,022 | 0.49431 / - | 0.49501 / - | +0.00070 ± 0.00084 (+0.00142, -0.00002) | +0.00002 ± 0.00098 | not scored | not scored | |"),
 ("the witness at @last for u20c (the @last column of C - A), the fold-1 controls W1S / W1L for u20b and u20c, the val / test domain-token convention ('rl': random-like one-hot instead of the training token) for u20b and u20c. Short witness feeds are scored with the training token [1, 0, 0]; val and test short feeds get [0, 1, 0]. The command in section 7 completes these.",
  "the witness at @last for u20c (the @last column of C - A), the fold-1 controls W1S / W1L for u20b and u20c, and the val / test domain-token convention ('rl': random-like one-hot instead of the training token) on long feeds and for u20c. The short witness was read under both tokens for A and B (training token [1, 0, 0]: +0.00060 ± 0.00082; val / test token [0, 1, 0]: +0.00070 ± 0.00084; difference of the delta +0.00010 ± 0.00061) [M]. The command in section 7 completes the rest."),
 ("$PY $D/aa_null_u20_k.py > $D/out/aa_null_u20_k.txt    # A/A null of the statistic, 2 and 4 seeds per arm",
  "$PY $D/wit_score.py rl W0S w5_e10_u20a_s951 w5_e10_u20a_s952 w5_e10_u20b_s951 w5_e10_u20b_s952; $PY $D/rl_short_read.py > $D/out/rl_short_read.txt   # short witness, val / test token\n$PY $D/aa_null_u20_k.py > $D/out/aa_null_u20_k.txt    # A/A null of the statistic, 2 and 4 seeds per arm"),
 ("`aa_null_u20.py`, `aa_null_u20_k.py`, `u20_interim_logs.py`, `recheck_cache.py`,", "`rl_short_read.py`, `aa_null_u20.py`, `aa_null_u20_k.py`, `u20_interim_logs.py`, `recheck_cache.py`,"),
 ("`cache/subxh/`, `cache/subxh_meta.npz`, `cache/scxh/`, `cache/marks/`.", "`cache/subxh/`, `cache/subxh_meta.npz`, `cache/scxh/`, `cache/marks/`, `cache/patch_report_*.py` (helpers that filled numbers into this report)."),
 ("`u20_wit_read_tr.txt`, `xh_read.txt`, `u20_read_sens.txt`,", "`u20_wit_read_tr.txt`, `xh_read.txt`, `rl_short_read.txt`, `u20_read_sens.txt`,"),
 ("3. **Transductive witness** (exploration-like train feeds of fold-0 users: unlabelled inputs of B, never seen by A; CPU vs CPU, selected): B - A = **+0.00060 ± 0.00082 on short feeds** (M < 50, n 11,022)",
  "3. **Transductive witness** (exploration-like train feeds of fold-0 users: unlabelled inputs of B, never seen by A; CPU vs CPU, selected): B - A = **+0.00060 ± 0.00082 on short feeds** (M < 50, n 11,022; +0.00070 ± 0.00084 with the val / test domain token)"),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s); print('patched', len(R))
