import re
p = '<repo>/reports/NEXT_1115/u20wit.md'
s = open(p).read()
R = [
 ("«XH_CA»", "+0.00100 ± 0.00069 (+0.00142, +0.00058); 2-pool -0.00002 ± 0.00077"),
 ("| CPU minus GPU-file value of the XH delta itself (same 8,000 feeds) | +0.00007 ± 0.00019 |",
  "| CPU minus GPU-file value of the XH delta itself (same 8,000 feeds) | +0.00007 ± 0.00019 |\n| C - A: H0 minus XH (both out-of-sample for C) | -0.00062 ± 0.00107 |\n| C - A: W0L minus XH, W0S minus XH (in-sample margin of C net of its data slope; bound only) | +0.01680 ± 0.00081, +0.02186 ± 0.00112 |"),
 ("On H0 the B - A member delta is -0.00086 on the CPU and -0.00111 from the GPU files: the CPU / GPU difference of a delta is about 0.00025, which is why only CPU numbers are compared with CPU numbers above.",
  "On the 8,000 XH feeds the six members read CPU minus GPU file -0.00006 .. -0.00017 (each within 1 SE). On H0 the B - A member delta is -0.00086 on the CPU and -0.00111 from the GPU files, on XH +0.00082 and +0.00075: the CPU / GPU difference of a delta is 0.0001-0.0003, which is why only CPU numbers are compared with CPU numbers above."),
 ("Written 2026-11-15 by the u20wit agent (CPU only; resumed run 01:08-04:00 UTC).", "Written 2026-11-15 03:50 UTC by the u20wit agent (CPU only; resumed run 01:08-03:50 UTC)."),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, 'w').write(s); print('left:', sorted(set(re.findall('«[A-Z0-9_]+»', s))), 'lines', s.count('\n'))
