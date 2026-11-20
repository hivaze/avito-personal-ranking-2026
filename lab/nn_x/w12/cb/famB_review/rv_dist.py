"""famB_review: label-free distribution of the 70 B3 columns on TRAIN rows (xm, fresh and lag-built) vs holdout / val / test rows (own code; 400K random rows per set).
shift = (median_set - median_xm) / (p90_xm - p10_xm); dNaN = NaN share minus the xm NaN share. Prints the columns with the largest train / apply mismatch."""
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
FEAT = "<repo>/nn_x/w12/cb/famB/feat"; HERE = "<repo>/nn_x/w12/cb/famB_review"
rg = np.random.default_rng(5); out = {}
for b in "abc":
    names = json.load(open(f"{FEAT}/{b}_names.json")); st = {}
    for s in ("xm", "lag:xm", "hold", "val", "test"):
        f = f"{FEAT}/lag_{b}_xm.npy" if s == "lag:xm" else f"{FEAT}/{b}_{s}.npy"
        a = np.load(f, mmap_mode="r"); idx = np.sort(rg.choice(a.shape[0], 400000, replace=False)); x = np.asarray(a[idx], np.float64)
        st[s] = dict(nan=np.isnan(x).mean(0), q=np.nanquantile(x, [0.1, 0.5, 0.9], axis=0))
    for j, n in enumerate(names):
        r = {}; q0 = st["xm"]["q"][:, j]; rng_ = max(q0[2] - q0[0], 1e-9)
        for s in st:
            q = st[s]["q"][:, j]; r[s] = dict(nan=float(st[s]["nan"][j]), p10=float(q[0]), p50=float(q[1]), p90=float(q[2]), shift=float((q[1] - q0[1]) / rng_), dnan=float(st[s]["nan"][j] - st["xm"]["nan"][j]),
                                           spread=float((q[2] - q[0]) / rng_))
        out[n] = r
json.dump(out, open(f"{HERE}/out/rv_dist.json", "w"), indent=1)
print(f"{'column':18s} | xm NaN p10/p50/p90 | val: dNaN shift spread | test: dNaN shift spread | hold: dNaN shift | lag-xm: dNaN shift")
key = lambda n: max(abs(out[n]["val"]["shift"]), abs(out[n]["test"]["shift"]), abs(out[n]["val"]["spread"] - 1), abs(out[n]["test"]["spread"] - 1))
for n in sorted(out, key=key, reverse=True):
    r = out[n]
    print(f"{n:18s} | {r['xm']['nan']:.3f} {r['xm']['p10']:.4g}/{r['xm']['p50']:.4g}/{r['xm']['p90']:.4g} | {r['val']['dnan']:+.3f} {r['val']['shift']:+.2f} {r['val']['spread']:.2f} | "
          f"{r['test']['dnan']:+.3f} {r['test']['shift']:+.2f} {r['test']['spread']:.2f} | {r['hold']['dnan']:+.3f} {r['hold']['shift']:+.2f} | {r['lag:xm']['dnan']:+.3f} {r['lag:xm']['shift']:+.2f}")
