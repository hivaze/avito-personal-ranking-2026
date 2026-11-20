"""Val per-feed NDCG of each pool (final 0.5 z(ft) + 0.5 z(to)) and of its ft / to halves.
Also label-free pool-vs-pool agreement on val / public / private (per-feed Spearman between pool scores)."""
from shiftlib import *  # noqa: F401,F403

fv, Mv, ov = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
ft_, Mt, ot = feed_meta("test"); pub = ft_["timestamp"].to_numpy() < BOUND
res, sc_val, sc_test = {}, {}, {}
for name in ["Cd", "PF7x", "G3", "M2", "Pall", "F7x", "F7b", "F7", "F7x70"]:
    if name in ("Cd", "PF7x", "G3", "M2", "Pall"):
        ftv, tov = np.load(P / f"ens_{name}_ft_val.npy"), np.load(P / f"ens_{name}_to_val.npy")
        ftt, tot = np.load(P / f"ens_{name}_ft_test.npy"), np.load(P / f"ens_{name}_to_test.npy")
        sv = 0.5 * zfeed(ftv, Mv) + 0.5 * zfeed(tov, Mv); st = 0.5 * zfeed(ftt, Mt) + 0.5 * zfeed(tot, Mt)
    else:
        sv, ftv, tov = split_pool(POOLS[name], "val", Mv)
        st, _, _ = split_pool(POOLS[name], "test", Mt)
    res[name] = ndcg_per_feed(sv.astype(np.float32), y, ov, Mv)
    res[name + "_ft"] = ndcg_per_feed(np.asarray(ftv, np.float32), y, ov, Mv)
    res[name + "_to"] = ndcg_per_feed(np.asarray(tov, np.float32), y, ov, Mv)
    sc_val[name], sc_test[name] = sv.astype(np.float32), st.astype(np.float32)
    print(f"{name:6s} val {res[name].mean():.5f}  ft {res[name + '_ft'].mean():.5f}  to {res[name + '_to'].mean():.5f}")
np.savez(OUT / "p2_val_perfeed.npz", **res)

print("\n== val deltas (paired) and ft / to halves ==")
for a, b in [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "PF7x"), ("G3", "PF7x"), ("F7x", "F7b"), ("F7x", "F7")]:
    print(f"{a:5s}-{b:5s} final {fmt(*mse(res[a] - res[b]))} | ft {fmt(*mse(res[a + '_ft'] - res[b + '_ft']))} | "
          f"to {fmt(*mse(res[a + '_to'] - res[b + '_to']))}")

# public-LB noise: SE of a paired delta on 8,562 public feeds, from the val per-feed delta spread
print("\n== expected public-LB noise of paired deltas (val per-feed SD / sqrt(n_public)) ==")
npub = int(pub.sum()); npriv = int((~pub).sum())
PUBLIC = {"PF7x": 0.392723, "G3": 0.392819, "M2": 0.392434, "Cd": 0.39208}
for a, b in [("PF7x", "Cd"), ("G3", "Cd"), ("M2", "Cd"), ("M2", "PF7x"), ("G3", "PF7x"), ("G3", "M2")]:
    d = res[a] - res[b]; sd = d.std()
    pv = PUBLIC[a] - PUBLIC[b]
    se_pub = sd / np.sqrt(npub); se_val = sd / np.sqrt(len(d))
    print(f"{a:5s}-{b:5s} val {d.mean():+.5f} (SE {se_val:.5f}) | public {pv:+.5f} (SE~{se_pub:.5f}) | "
          f"public-val {pv - d.mean():+.5f}, z {(pv - d.mean()) / np.hypot(se_pub, se_val):+.2f} | SE private~{sd / np.sqrt(npriv):.5f}")


def spear(a, b, M, off):
    from scipy.stats import rankdata
    n = len(M); fid = np.repeat(np.arange(n), M)
    # per-feed ranks via lexsort
    def frank(x):
        o = np.lexsort((x, fid)); r = np.empty(len(x)); r[o] = np.arange(len(x)) - np.repeat(off, M)
        return r
    ra, rb = frank(a), frank(b)
    za, zb = zfeed(ra, M), zfeed(rb, M)
    return np.bincount(fid, za * zb, n) / M


print("\n== label-free per-feed Spearman between pool scores: val | public | private ==")
rows = []
for a, b in [("M2", "PF7x"), ("G3", "PF7x"), ("M2", "Cd"), ("PF7x", "Cd"), ("F7x", "F7b"), ("F7x", "F7")]:
    rv = spear(sc_val[a], sc_val[b], Mv, ov); rt = spear(sc_test[a], sc_test[b], Mt, ot)
    print(f"{a:5s}~{b:5s} val {np.mean(rv):.4f} (1-rho {1 - np.mean(rv):.4f}) | public {np.mean(rt[pub]):.4f} "
          f"(1-rho {1 - np.mean(rt[pub]):.4f}) | private {np.mean(rt[~pub]):.4f} (1-rho {1 - np.mean(rt[~pub]):.4f})")
    rows.append(dict(cmp=f"{a}~{b}", val=float(np.mean(rv)), pub=float(np.mean(rt[pub])), priv=float(np.mean(rt[~pub]))))
    np.save(OUT / f"rho_{a}_{b}_val.npy", rv.astype(np.float32)); np.save(OUT / f"rho_{a}_{b}_test.npy", rt.astype(np.float32))
pl.DataFrame(rows).write_csv(OUT / "p2_pool_spearman.csv")
