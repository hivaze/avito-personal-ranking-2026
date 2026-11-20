"""Quick tests of set-level re-ranking rules on top of a base prediction (default nn_final2_val_oof), paired per-feed
NDCG deltas on the non-leak val feeds. Also a 2-fold (even/odd feed) selection check against grid-search optimism."""
import argparse, time
from fs_common import *
import rules as R

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="nn_final2_val_oof")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

t0 = time.time()
d = load_val()
X, miss = load_emb_norm(d["emb_idx"])
M, off, t, fid = d["M"], d["off"], d["target"], d["fid"]
use = ~d["leak"]
npos = np.bincount(fid, t, len(M))
s = np.load(PREDS / f"{args.base}.npy").astype(np.float64)
z = zfeed(s, off, M)
cat = d["category_id"].astype(np.int64)
base = ndcg_per_feed(z, t, off, M)
print(f"base {args.base}: all {base.mean():.5f}  non-leak {base[use].mean():.5f}  ({time.time()-t0:.1f}s)")

# top-1 precision
top1_pos = np.zeros(len(M), bool)
for f in range(len(M)):
    o, m = off[f], M[f]
    top1_pos[f] = t[o + np.argmax(z[o:o + m])] == 1
print(f"P(top1 positive) non-leak {top1_pos[use].mean():.4f}; among npos>=2 {top1_pos[use & (npos>=2)].mean():.4f}")

variants = []
for k in (1, 3, 5, 10):
    for a in (-0.5, -0.2, 0.2, 0.5, 1.0):
        variants.append((f"simprop_top{k}_a{a}", lambda k=k, a=a: R.simprop_topk(X, miss, off, M, z, k, a)))
for tau in (0.5, 1.0):
    for a in (-1.0, -0.5, 0.5, 1.0, 2.0):
        variants.append((f"softprop_tau{tau}_a{a}", lambda tau=tau, a=a: R.softprop(X, miss, off, M, z, tau, a, True)))
for thr in (0.8, 0.9, 0.95):
    for b in (0.1, 0.25, 0.5):
        variants.append((f"knnsmooth_thr{thr}_b{b}", lambda thr=thr, b=b: R.knn_smooth(X, miss, off, M, z, thr, b)))
for thr in (0.9, 0.95, 0.98):
    for b in (0.1, 0.25, 0.5):
        variants.append((f"dupmax_thr{thr}_b{b}", lambda thr=thr, b=b: R.dup_max(X, miss, off, M, z, thr, b)))
for a in (0.05, 0.1, 0.2, 0.5, -0.1):
    variants.append((f"mmr_a{a}", lambda a=a: R.mmr(X, miss, off, M, z, a)))
for tau in (0.5, 1.0):
    for a in (-0.2, 0.1, 0.2, 0.5):
        variants.append((f"catmass_tau{tau}_a{a}", lambda tau=tau, a=a: R.cat_mass(off, M, z, cat, tau, a, 0)))
for a in (-0.2, 0.1, 0.2, 0.5):
    variants.append((f"cattop1_a{a}", lambda a=a: R.cat_mass(off, M, z, cat, 1.0, a, 1)))
if args.quick:
    variants = variants[:3]

even = (np.arange(len(M)) % 2 == 0)
rows = []
for name, fn in variants:
    s2 = fn()
    p = ndcg_per_feed(s2, t, off, M)
    dd = p - base
    mu, se, n = paired(p, base, use)
    r = dict(name=name, delta=mu, se=se, t=mu / se if se > 0 else 0.0,
             d_top1pos=float(dd[use & top1_pos].mean()), d_top1neg=float(dd[use & ~top1_pos].mean()),
             d_multi=float(dd[use & (npos >= 2)].mean()), d_single=float(dd[use & (npos == 1)].mean()),
             d_even=float(dd[use & even].mean()), d_odd=float(dd[use & ~even].mean()),
             changed=float((dd[use] != 0).mean()))
    rows.append(r)
    print(f"{name:26s} d={mu:+.5f} ± {se:.5f} (t={r['t']:+.1f}) | top1+ {r['d_top1pos']:+.5f} top1- {r['d_top1neg']:+.5f} | multi {r['d_multi']:+.5f} single {r['d_single']:+.5f} | changed {r['changed']:.3f}", flush=True)

# 2-fold selection: choose best variant on even feeds, report on odd, and vice versa
be = max(rows, key=lambda r: r["d_even"]); bo = max(rows, key=lambda r: r["d_odd"])
cv = 0.5 * (be["d_odd"] + bo["d_even"])
print(f"\n2-fold selection: best-on-even {be['name']} -> odd {be['d_odd']:+.5f}; best-on-odd {bo['name']} -> even {bo['d_even']:+.5f}; CV estimate {cv:+.5f}")
json.dump(dict(base=args.base, base_nonleak=float(base[use].mean()), rows=rows, cv=cv, sel=[be["name"], bo["name"]]),
          open(OUT / f"out_02_rerank_{args.base}.json", "w"), indent=1)
print(f"done {time.time()-t0:.1f}s")
