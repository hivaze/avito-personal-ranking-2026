p = '<repo>/reports/NEXT_1115/u20wit.md'
s = open(p).read()
R = [
 ("| XH = cross-user holdout, random 8,000 of 24,026 (inputs and users' labels: seen by nobody / by both) |",
  "| XH = cross-user holdout, random 8,000 of 24,026 (feeds in nobody's training; their users are labelled for A and B alike) |"),
 ("- Light read-outs (2 threads, under 2 minutes each) ran next to my one heavy scoring process several times.",
  "- Rule 1, stated plainly: heavy jobs (12 threads) ran strictly one at a time with CUDA_VISIBLE_DEVICES=\"\"; light read-outs (2 threads, under 2 minutes each) ran next to the one heavy scoring process several times; a few python one-liners that only edited text files of mine (PROGRESS.md, this report; no torch, no data) were started without the variable. Nothing touched a GPU."),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s); print('patched', len(R))
p = '<repo>/nn_x/w12/next/u20wit/PROGRESS.md'
s = open(p).read()
i = s.index('## Running processes of mine'); j = s.index('- stage1.sh FINISHED')
s = s[:i] + """## Running processes of mine (state 03:52 UTC): NONE. The task is COMPLETE.
- stage2d.sh (PID 4060379) printed STAGE2D DONE at 03:45:17; wit_score.py rl W0S for u20b (PID 4104422) ended 03:49:06.
- stopped earlier by exact PID (all mine): stage2.sh 3910465, stage2b.sh 3933450, stage2c.sh 4028353 + xh_score 4043591, stage2c.sh 4044198.
- NOT run for lack of time: @last of u20c on the witness, controls W1S / W1L for u20b / u20c, the rl token on long feeds and for u20c
  (`SEEDS=951,952 bash nn_x/w12/next/u20wit/run_seeds.sh` completes them; idempotent).
""" + s[j:]
i = s.index('## Next')
s = s[:i] + """## FINAL (03:52 UTC)
- report: reports/NEXT_1115/u20wit.md (final, no placeholders). DECISION by the fixed rule on the cross-user columns: 2 MORE SEEDS.
- XH (CPU, 8,000 cross-user holdout feeds; out/xh_read.txt): B-A +0.00082+-0.00067, C-A +0.00100+-0.00069; excess of the witness over it: W0L +0.00013+-0.00076, W0S -0.00022+-0.00106
- short witness with the val / test token (out/rl_short_read.txt): B-A +0.00070+-0.00084 (tr token +0.00060+-0.00082)
- for the 4-seed read: SEEDS=951,952,953,954 WAIT=1 bash nn_x/w12/next/u20wit/run_seeds.sh (xh: add `xh_score.py <new variants>` then `xh_read.py --seeds 951,952,953,954`)
"""
open(p, 'w').write(s); print('progress updated')
