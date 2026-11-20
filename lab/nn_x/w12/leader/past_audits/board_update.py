"""Board arithmetic for the past-audits role (2026-11-18). CPU only, no data read.
Sources of the inputs: docs/05-timeline.md (board snapshots), docs/07-what-failed.md (14 files of the final family:
mean 0.39480, sd 0.00021, max 0.3952092), lab/nn_x/audit/top1/findings.txt (cross-team paired SE 0.0012-0.0018,
near-copy paired SE 0.0005-0.0007)."""
import numpy as np

lead = 0.3982556
b13 = {"t": "13 Nov 19:43", "r2": 0.395115, "r3": 0.393778, "us": 0.393206}
b18 = {"t": "18 Nov 03:33", "r2": 0.3972749, "r3": 0.3964761, "r4": 0.3960596, "r5": 0.3958928,
       "r6": 0.3954462, "r7": 0.3953650, "us": 0.3952092}
se_x = (0.0012, 0.0018)

print("leader - 2nd: 13 Nov %+.5f | 18 Nov %+.5f" % (lead - b13["r2"], lead - b18["r2"]))
print("leader - us : 12 Nov 22:00 %+.5f | 13 Nov %+.5f | 18 Nov %+.5f" % (lead - 0.3928199, lead - b13["us"], lead - b18["us"]))
print("2nd-best score rise 13->18 Nov %+.5f ; our rise %+.5f ; 3rd-best rise %+.5f" %
      (b18["r2"] - b13["r2"], b18["us"] - b13["us"], b18["r3"] - b13["r3"]))
for k in ("r2", "r3", "r4", "r5", "r6", "r7"):
    d = b18[k] - b18["us"]
    print("18 Nov %s - us %+.5f = %.1f-%.1f cross-team SE" % (k, d, d / se_x[1], d / se_x[0]))
d = lead - b18["r2"]
print("leader - 2nd = %.2f-%.2f cross-team SE" % (d / se_x[1], d / se_x[0]))
gaps = np.diff(sorted([lead] + [b18[k] for k in ("r2", "r3", "r4", "r5", "r6", "r7", "us")]))[::-1]
print("consecutive gaps top-8 (desc rank):", " ".join("%.5f" % g for g in gaps))

# best-of-K selection optimism on public among near-copies (own data: K=14, sd 0.00021, max-mean +0.00041)
rng = np.random.default_rng(0)
for sd in (0.00021, 0.0003, 0.0005):
    row = []
    for K in (3, 14, 30, 60, 100):
        m = rng.standard_normal((200000, K)).max(1).mean() * sd
        row.append("K=%d +%.5f" % (K, m))
    print("E[max]-mean, sd %.5f: " % sd + " | ".join(row))
print("own final family: max - mean = %+.5f (K=14)" % (0.3952092 - 0.39480))
