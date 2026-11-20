"""CHECK 2 / 3 / 4: my nested residual read (direct-NDCG models D1, D2; my folds), my placebo, position slices.
Usage: r3_nested.py            -> out/r3_nested.json, out/r3_perfeed.npz
"""
import json
import sys
import time

import numpy as np

from rlib import OUT, ROOT, Val, ndcg_feeds, ndcg_grid

t0 = time.time()
V = Val()
print(f"pool score b (my zf, my NDCG): plain val NDCG {V.base.mean():.6f}", flush=True)
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # cross-check of my metric only
import pmix

ref = ndcg_per_feed(V.b, V.y, V.off, V.M)
rng0 = np.random.default_rng(1)
pert = V.b + 0.3 * np.round(rng0.normal(size=V.N), 1)   # a score with many ties
print(f"metric cross-check: max per-feed |mine - src/metrics| on b {np.abs(ref - V.base).max():.2e}; on a tied score "
      f"{np.abs(ndcg_per_feed(pert, V.y, V.off, V.M) - ndcg_feeds(pert, V.y, V.off, V.M)).max():.2e}", flush=True)
assert abs(V.base.mean() - 0.40271) < 2e-5
C = np.load(f"{OUT}/my_counts.npy")[:V.N].astype(np.float64)   # a, a24, b, d, e, a_val, a_test
GRID = np.array(sorted([0.0] + [s * v for s in (1, -1) for v in (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)]))
G0 = int(np.flatnonzero(GRID == 0.0)[0])
FOLD_SEEDS = (11, 12, 13)
cold = V.cold.astype(np.float64)


def feats(Cx, which):
    """which = 'A' (as-of windows a, a24) or 'L' (later-using windows d, b, e). Returns (names, [columns]); the last two are
    the b-interaction columns used by D2 only."""
    if which == "A":
        n1, n2 = Cx[:, 0], Cx[:, 1]
        r1, z1 = V.forms(n1); _, z2 = V.forms(n2)
        i1 = (n1 >= 1).astype(np.float64); i2 = (n2 >= 1).astype(np.float64)
        names = ["z_a", "r_a-0.5", "i_a", "cs_a", "l_a", "z_a24", "i_a24", "b*i_a", "b*cs_a"]
        cols = [z1, r1 - 0.5, i1, cold * i1, np.log1p(n1), z2, i2, V.b * i1, V.b * cold * i1]
        return names, cols, 7
    nd, nb_, ne = Cx[:, 3], Cx[:, 2], Cx[:, 4]
    rd, zd = V.forms(nd); _, zb = V.forms(nb_); _, ze = V.forms(ne)
    idd = (nd >= 1).astype(np.float64); ib = (nb_ >= 1).astype(np.float64); ie = (ne >= 1).astype(np.float64)
    names = ["z_d", "r_d-0.5", "i_d", "cs_d", "l_d", "z_b", "i_b", "z_e", "i_e", "b*i_d", "b*cs_d"]
    cols = [zd, rd - 0.5, idd, cold * idd, np.log1p(nd), zb, ib, ze, ie, V.b * idd, V.b * cold * idd]
    return names, cols, 9


def d1_tables(cols, n1):
    P = np.empty((n1, len(GRID), V.nf))
    for j in range(n1):
        ndcg_grid(V.b, np.ascontiguousarray(cols[j]), GRID, V.y, V.off, V.M, P[j])
    return P


def d1_nested(P, fold):
    out = np.empty(V.nf); picks = []
    for k in range(5):
        tr = fold != k
        mt = P[:, :, tr].mean(2)
        j, g = np.unravel_index(np.argmax(mt), mt.shape)
        if mt[j, g] <= mt[0, G0] + 1e-7:
            j, g = 0, G0
        out[~tr] = P[j, g][~tr]; picks.append((int(j), float(GRID[g])))
    return out, picks


def d2_nested(cols, fold, passes=2):
    out = np.empty(V.nf); ws = []
    buf = np.empty((len(GRID), V.nf))
    for k in range(5):
        tr = fold != k
        w = np.zeros(len(cols)); base = V.b.copy(); cur = V.base[tr].mean()
        for _ in range(passes):
            for j in range(len(cols)):
                bm = base - w[j] * cols[j]
                ndcg_grid(bm, np.ascontiguousarray(cols[j]), GRID, V.y, V.off, V.M, buf)
                mt = buf[:, tr].mean(1); g = int(np.argmax(mt))
                if mt[g] > cur + 1e-7:
                    w[j] = GRID[g]; base = bm + w[j] * cols[j]; cur = mt[g]
        out[~tr] = ndcg_feeds(base, V.y, V.off, V.M)[~tr]; ws.append(w.tolist())
    return out, ws


def read(Cx, which, folds, want_detail=False):
    names, cols, n1 = feats(Cx, which)
    P = d1_tables(cols, n1)
    d1 = []; d2 = []; picks = []; ws = []
    for fold in folds:
        o, p = d1_nested(P, fold); d1.append(o - V.base); picks.append(p)
        o, w = d2_nested(cols, fold); d2.append(o - V.base); ws.append(w)
    d1 = np.mean(d1, 0); d2 = np.mean(d2, 0)
    if want_detail:
        mt = P.mean(2) - V.base.mean()
        j, g = np.unravel_index(np.argmax(mt), mt.shape)
        ceil = dict(best_feature=names[j], best_w=float(GRID[g]), gain=float(mt[j, g]), se=float((P[j, g] - V.base).std() / np.sqrt(V.nf)),
                    per_feature_best={names[jj]: [float(GRID[int(np.argmax(mt[jj]))]), float(mt[jj].max())] for jj in range(n1)})
        return d1, d2, picks, ws, names, ceil
    return d1, d2


def st(d, w=None):
    if w is None:
        g = d.mean(); se = d.std() / np.sqrt(len(d))
    else:
        g = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum(); se = np.sqrt(np.average((d - g) ** 2, weights=w) / neff)
    return [float(g), float(se), float(g / se) if se > 0 else 0.0]


folds = [V.folds(s) for s in FOLD_SEEDS]
folds_u0g = [V.folds(s, u0_one_group=True) for s in FOLD_SEEDS]
for f in folds:
    # no user in two folds (non-zero users)
    nz = V.uid != 0
    import collections
    mp = collections.defaultdict(set)
    for u, k in zip(V.uid[nz], f[nz]):
        mp[int(u)].add(int(k))
    assert max(len(v) for v in mp.values()) == 1
print(f"folds built (5 x seeds {FOLD_SEEDS}); fold sizes seed {FOLD_SEEDS[0]}: {np.bincount(folds[0]).tolist()} | {time.time() - t0:.0f}s", flush=True)
W = pmix.weights()
res = {"base": float(V.base.mean()), "grid": GRID.tolist()}
save = {}
for which in ("A", "L"):
    d1, d2, picks, ws, names, ceil = read(C, which, folds, want_detail=True)
    res[which] = dict(D1=st(d1), D2=st(d2), D1_pub=st(d1, W["public"]), D1_priv=st(d1, W["private"]), D2_pub=st(d2, W["public"]), D2_priv=st(d2, W["private"]),
                      D1_picks=[[(names[j], w) for j, w in p] for p in picks], D2_w_mean=np.mean(np.array(ws).reshape(-1, len(names)), 0).tolist(),
                      D2_w_nonzero_share=(np.array(ws).reshape(-1, len(names)) != 0).mean(0).tolist(), names=names, ceiling_D1=ceil)
    save[f"d1_{which}"] = d1; save[f"d2_{which}"] = d2
    print(f"\n=== set {which}: REAL nested gain vs the RAW pool score (5 folds x 3 seeds, 14,423 feeds)")
    print(f"  D1 {res[which]['D1'][0]:+.5f} +- {res[which]['D1'][1]:.5f} (z {res[which]['D1'][2]:+.2f}) | pmix public {res[which]['D1_pub'][0]:+.5f}+-{res[which]['D1_pub'][1]:.5f} private {res[which]['D1_priv'][0]:+.5f}+-{res[which]['D1_priv'][1]:.5f}")
    print(f"  D2 {res[which]['D2'][0]:+.5f} +- {res[which]['D2'][1]:.5f} (z {res[which]['D2'][2]:+.2f}) | pmix public {res[which]['D2_pub'][0]:+.5f}+-{res[which]['D2_pub'][1]:.5f} private {res[which]['D2_priv'][0]:+.5f}+-{res[which]['D2_priv'][1]:.5f}")
    print(f"  D1 picks per fit (feature, w): {res[which]['D1_picks']}")
    print("  D2 mean w / share of fits with w != 0: " + ", ".join(f"{n} {m:+.3f}/{s:.2f}" for n, m, s in zip(names, res[which]['D2_w_mean'], res[which]['D2_w_nonzero_share'])))
    print(f"  IN-SAMPLE ceiling of D1 (fit = score on all feeds): best {ceil['best_feature']} w {ceil['best_w']:+.3f} gain {ceil['gain']:+.5f} +- {ceil['se']:.5f}; "
          + ", ".join(f"{k} {v[1]:+.5f}@{v[0]:+.3f}" for k, v in ceil['per_feature_best'].items()), flush=True)
    if which == "A":
        d1g, d2g = read(C, which, folds_u0g)
        res["A_U0G"] = dict(D1=st(d1g), D2=st(d2g))
        print(f"  variant U0G (all uid-0 feeds in one fold group): D1 {st(d1g)[0]:+.5f} +- {st(d1g)[1]:.5f} | D2 {st(d2g)[0]:+.5f} +- {st(d2g)[1]:.5f}")
        # slices (CHECK 4)
        cov = np.bincount(V.fid, C[:, 0] >= 1, V.nf) / V.M
        tod = (V.ts + 10800) % 86400
        third = np.zeros(V.nf, np.int64)
        for d in (20718, 20720):
            m = V.day == d
            third[m] = np.digitize(tod[m], np.quantile(tod[m], [1 / 3, 2 / 3]))
        qc = np.digitize(cov, np.quantile(cov, [0.2, 0.4, 0.6, 0.8]))
        cs = np.bincount(V.fid, V.cold, V.nf) / V.M
        SL = {"day20718": V.day <= 20719, "day20720": V.day >= 20720}
        SL.update({f"d{d % 100}:third{t + 1}": (V.day == d) & (third == t) for d in (20718, 20720) for t in range(3)})
        SL.update({f"covQ{q + 1} (mean cov {100 * cov[qc == q].mean():.1f}%)": qc == q for q in range(5)})
        SL.update({"cold<30%": cs < 0.30, "cold30-45%": (cs >= 0.30) & (cs < 0.45), "cold>=45%": cs >= 0.45, "uid0": V.uid == 0, "uid!=0": V.uid != 0})
        res["A_slices"] = {}
        print("  slices of the per-feed gains (D1 | D2):")
        for k, m in SL.items():
            res["A_slices"][k] = dict(n=int(m.sum()), D1=st(d1[m]), D2=st(d2[m]))
            print(f"    {k:28s} [{int(m.sum()):5d}] D1 {st(d1[m])[0]:+.5f}+-{st(d1[m])[1]:.5f} | D2 {st(d2[m])[0]:+.5f}+-{st(d2[m])[1]:.5f}")
        m1, m2 = SL["day20718"], SL["day20720"]
        for nm, d in (("D1", d1), ("D2", d2)):
            dif = d[m2].mean() - d[m1].mean(); sed = np.sqrt(d[m2].var() / m2.sum() + d[m1].var() / m1.sum())
            res[f"A_day_diff_{nm}"] = [float(dif), float(sed)]
            print(f"    {nm}: day 20720 minus day 20718 = {dif:+.5f} +- {sed:.5f} (z {dif / sed:+.2f})")
np.savez(f"{OUT}/r3_perfeed.npz", **save)
json.dump(res, open(f"{OUT}/r3_nested.json", "w"), indent=1)
print(f"real reads done {time.time() - t0:.0f}s", flush=True)

# ---------------- placebos
order_orig_cache = {}


def permute_within(grp, rng):
    """A random permutation of the rows inside each group: returns src so that placebo = C[src]."""
    u = rng.random(V.N)
    so = np.argsort(grp, kind="stable")
    sr = np.lexsort((u, grp))
    src = np.empty(V.N, np.int64); src[so] = sr
    return src


grp_feed = V.fid * 2 + V.cold.astype(np.int64)
blk = (((V.ts + 10800) % 86400) // 10800)[V.fid]
grp_strat = (V.day[V.fid] - V.day.min()) * 16 + blk * 2 + V.cold.astype(np.int64)
pl_out = open(f"{OUT}/r3_placebo.jsonl", "w")
for kind, grp, seeds in (("WF", grp_feed, range(500, 520)), ("STRAT", grp_strat, range(600, 610))):
    for seed in seeds:
        src = permute_within(grp, np.random.default_rng(seed))
        Cp = C[src]
        assert (np.bincount(grp, Cp[:, 0]) == np.bincount(grp, C[:, 0])).all()
        rec = {"kind": kind, "seed": seed, "corr_a": float(np.corrcoef(Cp[:, 0], C[:, 0])[0, 1]), "same_a": float((Cp[:, 0] == C[:, 0]).mean())}
        for which in ("A", "L"):
            d1, d2 = read(Cp, which, folds)
            rec[which] = dict(D1=st(d1), D2=st(d2))
        pl_out.write(json.dumps(rec) + "\n"); pl_out.flush()
        print(f"[{kind}] seed {seed}: A D1 {rec['A']['D1'][0]:+.5f} D2 {rec['A']['D2'][0]:+.5f} | L D1 {rec['L']['D1'][0]:+.5f} D2 {rec['L']['D2'][0]:+.5f} | corr(a) {rec['corr_a']:.3f} | {time.time() - t0:.0f}s", flush=True)
pl_out.close()
print(f"DONE {time.time() - t0:.0f}s")
