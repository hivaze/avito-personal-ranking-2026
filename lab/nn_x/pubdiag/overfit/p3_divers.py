"""Part 3: diversity on val / holdout / test(public, private): within-feed z correlations (feed-equal), same-root
correlations, effective number of members, ensemble gain (pool - mean member) on val and holdout.
Writes p3_divers.txt and cache/corr_*.npy."""
from lib import *

C = OUT / "cache"
idx = {m: i for i, m in enumerate(MEMBERS)}
nm = len(MEMBERS)
lines = []
P_ = lambda s="": (print(s, flush=True), lines.append(s))


def corr_matrix(Z, M, fid, mask=None):
    """Feed-equal mean of within-feed Pearson correlations between rows of Z (z-scored per feed)."""
    w = np.where(M >= 2, 1.0 / M, 0.0)
    if mask is not None:
        w = w * mask
    wr = np.sqrt(w[fid]).astype(np.float32)
    A = np.asarray(Z, np.float32) * wr[None, :]
    G = (A @ A.T).astype(np.float64)
    d = np.sqrt(np.diag(G))
    return G / np.outer(d, d)


Sv, Sh, St = Split("val"), Split("holdout"), Split("test")
pub = St.ts < PUB_TS
CM = {}
for split, S, halves, masks in [("val", Sv, ("to", "ft"), {"": None}), ("holdout", Sh, ("to",), {"": None}),
                                ("test", St, ("to", "ft"), {"": None, "pub": pub, "priv": ~pub})]:
    Zs = [np.load(C / f"Z_{split}_{h}.npy") for h in halves]
    Z = np.concatenate(Zs, 0)
    for mk, mask in masks.items():
        key = split + ("_" + mk if mk else "")
        f = C / f"corr_{key}.npy"
        if f.exists():
            CM[key] = np.load(f)
        else:
            CM[key] = corr_matrix(Z, S.M, S.fid, mask)
            np.save(f, CM[key])
        print("corr", key, CM[key].shape, flush=True)
    del Z, Zs

KEYS = ["val", "holdout", "test", "test_pub", "test_priv"]


def sub(key, ms, half="to", half2=None):
    off = {"to": 0, "ft": nm}
    a = [idx[m] + off[half] for m in ms]
    b = [idx[m] + off[half2 or half] for m in ms]
    return CM[key][np.ix_(a, b)]


def rbar(R):
    n = len(R)
    return (R.sum() - np.trace(R)) / (n * (n - 1))


def stats(R):
    n = len(R)
    rb = rbar(R)
    ev = np.linalg.eigvalsh(R)
    return rb, n / (1 + (n - 1) * rb), ev.sum() ** 2 / (ev ** 2).sum()


P_("== Part 3a: same-root correlations (feed-equal within-feed z corr, TO preds)")
pairs = [("F7x", "F7x70"), ("F7x", "F7"), ("F7x", "F7b"), ("F7x70", "F7"), ("F7b", "F7"), ("F7x70", "F7b")]
P_("pair            " + "  ".join(f"{k:>9s}" for k in KEYS) + "   | cross-root same pair (val, test)")
for a, b in pairs:
    vals = []
    for k in KEYS:
        vals.append(np.mean([CM[k][idx[FAM[a][i]], idx[FAM[b][i]]] for i in range(7)]))
    cr = [np.mean([CM[k][idx[FAM[a][i]], idx[FAM[b][j]]] for i in range(7) for j in range(7) if i != j])
          for k in ("val", "test")]
    P_(f"{a:>6s}~{b:<7s}  " + "  ".join(f"{v:9.4f}" for v in vals) + f"   | {cr[0]:.4f} {cr[1]:.4f}")
P_("within-family cross-seed mean corr (TO):")
for f in FAM:
    P_(f"  {f:6s} " + "  ".join(f"{k} {rbar(sub(k, FAM[f])):.4f}" for k in KEYS)
       + (f" | FT val {rbar(sub('val', FAM[f], 'ft')):.4f} test {rbar(sub('test', FAM[f], 'ft')):.4f}"))

P_()
P_("== Part 3b: pool diversity: mean pairwise corr rho, n_eff = n/(1+(n-1)rho), participation ratio PR")
for half in ("to", "ft"):
    P_(f"-- half {half}")
    for p in POOLS:
        ms = pool_members(p)
        s = f"  {p:5s} n={len(ms):2d} "
        for k in KEYS:
            if half == "ft" and k == "holdout":
                continue
            rb, ne, pr = stats(sub(k, ms, half))
            s += f"| {k} rho {rb:.4f} neff {ne:5.2f} PR {pr:5.2f} "
        P_(s)
P_("-- test - val change in rho (positive = less diverse on test):")
for p in POOLS:
    ms = pool_members(p)
    P_(f"  {p:5s} TO {rbar(sub('test', ms)) - rbar(sub('val', ms)):+.4f} (pub {rbar(sub('test_pub', ms)) - rbar(sub('val', ms)):+.4f},"
       f" priv {rbar(sub('test_priv', ms)) - rbar(sub('val', ms)):+.4f}) | FT {rbar(sub('test', ms, 'ft')) - rbar(sub('val', ms, 'ft')):+.4f}"
       f" | holdout-val TO {rbar(sub('holdout', ms)) - rbar(sub('val', ms)):+.4f}")
P_("-- corr between a member's own TO and FT preds (mean over members), and between pool TO-mean and FT-mean:")
for p in POOLS:
    ms = pool_members(p)
    own = {k: np.mean(np.diag(sub(k, ms, "to", "ft"))) for k in ("val", "test", "test_pub", "test_priv")}
    P_(f"  {p:5s} own TO~FT " + " ".join(f"{k} {v:.4f}" for k, v in own.items()))

# pool-level TO-mean vs FT-mean correlation and cross-pool correlations (needs pool vectors)
P_()
P_("== Part 3c: correlation of pool vectors (z-mean of members), val vs test(pub/priv)")
pv = {}
for split, S, masks in [("val", Sv, {"": None}), ("test", St, {"pub": pub, "priv": ~pub})]:
    Zto = np.load(C / f"Z_{split}_to.npy", mmap_mode="r"); Zft = np.load(C / f"Z_{split}_ft.npy", mmap_mode="r")
    vec = {}
    for p in POOLS:
        ii = [idx[m] for m in pool_members(p)]
        to = S.z(np.asarray(Zto[ii], np.float64).mean(0)); ft = S.z(np.asarray(Zft[ii], np.float64).mean(0))
        vec[p + ":to"], vec[p + ":ft"], vec[p + ":full"] = to, ft, S.z(0.5 * to + 0.5 * ft)
    names = list(vec)
    Zp = np.stack([vec[n] for n in names]).astype(np.float32)
    for mk, mask in masks.items():
        pv[split + ("_" + mk if mk else "")] = (names, corr_matrix(Zp, S.M, S.fid, mask))
for p in POOLS:
    s = f"  {p:5s}"
    for k, (names, R) in pv.items():
        i, j = names.index(p + ":to"), names.index(p + ":ft")
        c = names.index("Cd:full"), names.index(p + ":full")
        s += f" | {k}: TOmean~FTmean {R[i, j]:.4f}, full~Cd {R[c]:.4f}"
    P_(s)

P_()
P_("== Part 3d: ensemble gain = pool z-mean NDCG - mean single-member NDCG")
Zv = {h: np.load(C / f"Z_val_{h}.npy", mmap_mode="r") for h in ("to", "ft")}
Zh = np.load(C / "Z_holdout_to.npy", mmap_mode="r")
single = {("val", h, m): Sv.nd(np.asarray(Zv[h][idx[m]])) for h in ("to", "ft") for m in MEMBERS}
single.update({("hold", "to", m): Sh.nd(np.asarray(Zh[idx[m]])) for m in MEMBERS})
EG = {}
for p in POOLS:
    ms = pool_members(p); ii = [idx[m] for m in ms]
    r = {}
    for nmk, S, Z, h in [("val_to", Sv, Zv["to"], "to"), ("val_ft", Sv, Zv["ft"], "ft"), ("hold_to", Sh, Zh, "to")]:
        pool_nd = S.nd(S.z(np.asarray(Z[ii], np.float64).mean(0)))
        mean_single = np.mean([single[("val" if nmk.startswith("val") else "hold", h, m)] for m in ms], 0)
        g = pool_nd - mean_single
        r[nmk] = (pool_nd.mean(), mean_single.mean(), g.mean(), se(g))
    EG[p] = r
    P_(f"  {p:5s} " + " | ".join(f"{k}: pool {v[0]:.5f} single {v[1]:.5f} gain {v[2]:+.5f}+-{v[3]:.5f}" for k, v in r.items()))
P_("ensemble gain per unit (1-rho) (TO half), val vs holdout; and the test-implied gain if gain scales with (1-rho):")
for p in POOLS:
    ms = pool_members(p)
    rv, rh, rt = rbar(sub("val", ms)), rbar(sub("holdout", ms)), rbar(sub("test", ms))
    rpub, rpriv = rbar(sub("test_pub", ms)), rbar(sub("test_priv", ms))
    gv, gh = EG[p]["val_to"][2], EG[p]["hold_to"][2]
    P_(f"  {p:5s} val gain/(1-rho) {gv / (1 - rv):.4f}  hold {gh / (1 - rh):.4f} | val gain {gv:+.5f} -> test-implied"
       f" {gv * (1 - rt) / (1 - rv):+.5f} (pub {gv * (1 - rpub) / (1 - rv):+.5f}, priv {gv * (1 - rpriv) / (1 - rv):+.5f})")
(OUT / "p3_divers.txt").write_text("\n".join(lines) + "\n")
