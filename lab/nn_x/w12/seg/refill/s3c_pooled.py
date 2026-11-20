"""Addendum to s3_read.py (run AFTER the first read; same statistics, holdout and val read sets pooled; descriptive, declared in
block 0 as 'pooled holdout + val'). Contrasts that s3_read.py printed per set only: c - e, c - am, e - b, d1 - a, d2 - a, dP - a,
d1 - dP, g - b192, and the count of members with a positive numerator."""
import numpy as np
import s3_read as R

S = {k: R.Set(k) for k in ("holdout", "val")}
print("## pooled holdout + val (feeds of both read sets, users as clusters) [M]")
for tag in "TU":
    for who in ("pool4", "distilled", "hard"):
        row = []
        for lab, pa, pb in (("c - b", "c" + tag, "b"), ("a - b", "a", "b"), ("am - b", "am" + tag, "b"), ("c - am", "c" + tag, "am" + tag), ("e - b", "e" + tag, "b"), ("c - e", "c" + tag, "e" + tag),
                            ("d1 - a", "d1" + tag, "a"), ("d2 - a", "d2" + tag, "a"), ("dP - a", "dP" + tag, "a"), ("d1 - dP", "d1" + tag, "dP" + tag)):
            parts = []
            for name in ("holdout", "val"):
                s = S[name]
                mask = s.read[tag] & ~np.isnan(s.ndcg(pa, who)) & ~np.isnan(s.ndcg(pb, who))
                parts.append((R.diff(s, pa, pb, who, mask), s.clu[mask]))
            x, cl = R.cat(parts)
            row.append(f"{lab} {R.f(*R.cstat(x, cl)[:2])}")
        print(f"   design {tag} {who:9s} n {len(x)}: " + " | ".join(row))
for lab in ("all capped",):
    parts_n, parts_d, parts_x = [], [], []
    for name in ("holdout", "val"):
        s = S[name]
        m = np.ones(s.n, bool)
        parts_n.append((R.diff(s, "g", "b", "pool4", m), s.clu))
        parts_d.append((R.diff(s, "b192", "b", "pool4", m), s.clu))
        parts_x.append((R.diff(s, "g", "b192", "pool4", m), s.clu))
    n, cl = R.cat(parts_n)
    d, _ = R.cat(parts_d)
    x, _ = R.cat(parts_x)
    lo, hi, _ = R.boot_ratio(n, d, cl)
    print(f"   calibration g, all capped feeds of both sets (n {len(n)}): g - b {R.f(*R.cstat(n, cl)[:2])} | b192 - b {R.f(*R.cstat(d, cl)[:2])} | rho_g {n.mean() / d.mean():+.3f} [boot 95% {lo:+.3f}, {hi:+.3f}] | g - b192 {R.f(*R.cstat(x, cl)[:2])}")
print("## members with a positive literal numerator c - b (design x set x member)")
for tag in "TU":
    for name in ("holdout", "val"):
        s = S[name]
        v = [R.diff(s, "c" + tag, "b", w, s.read[tag]).mean() for w in R.WHO if w.startswith("m:")]
        print(f"   design {tag} {name}: {sum(x > 0 for x in v)} of 4 positive ({', '.join(f'{x:+.5f}' for x in v)})")
