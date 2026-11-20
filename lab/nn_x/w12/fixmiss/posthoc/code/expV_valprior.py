"""(e2) Fresh item prior from OTHER users' validation clicks (for test: all val labels precede test; train-only and FT
branches never see item-level val click counts). Cross-fit: rows of user half h get statistics from half 1-h only
(never the same user). Two time regimes: 'any' (other-half feeds of any val day; optimistic, simultaneous) and
'earlier' (other-half feeds of strictly earlier days, the test-like direction; covers day-2 feeds only).
key = s + beta * x, x in {1[item clicked by another user], log1p(clicks), log((clicks+1)/(imps*p0+1))}; beta on half A, applied on B."""
import ph, numpy as np, json
D = ph.Data("val"); fid = D.fid; s = D.s
item = np.load("<repo>/data/flat/val/cand_item_id.npy").astype(np.int64)
ts = D.f["timestamp"].to_numpy(); day = ((ts - ts.min()) // 86400)[fid]
p0 = D.y.mean()
for regime in ("any", "earlier"):
    X = {k: np.zeros(len(s)) for k in ("hit", "log1p", "lift")}
    for h in (0, 1):
        src = D.halfr != h; dst = D.halfr == h
        for d in np.unique(day[dst]):
            sm = src & ((day < d) if regime == "earlier" else np.ones(len(s), bool))
            dm = dst & (day == d)
            u, inv = np.unique(item[sm], return_inverse=True)
            pos = np.bincount(inv, D.y[sm], len(u)); imp = np.bincount(inv, None, len(u))
            j = np.searchsorted(u, item[dm]); j = np.minimum(j, len(u) - 1) if len(u) else j; hit = (len(u) > 0) & (u[j] == item[dm]) if len(u) else np.zeros(dm.sum(), bool)
            cp = np.where(hit, pos[j] if len(u) else 0, 0); ci = np.where(hit, imp[j] if len(u) else 0, 0)
            X["hit"][dm] = (cp > 0).astype(float); X["log1p"][dm] = np.log1p(cp); X["lift"][dm] = np.where(ci > 0, np.log((cp + 1) / (ci * p0 + 1)), 0)
    cov = (X["hit"] > 0)
    print(f"[{regime}] rows with an other-user val click: {cov.mean():.4%}; positive rate there {D.y[cov].mean():.4f} vs {p0:.4f}; feeds touched {np.bincount(fid, cov, D.n).astype(bool).mean():.3%}")
    for nm, x in X.items():
        grid = np.round(np.arange(-0.2, 1.21, 0.1), 2)
        tab = np.stack([D.ev(s + b * x) for b in grid], 1)
        out = np.zeros(D.n)
        for h in (0, 1):
            A = D.half == h; bi = tab[A].mean(0).argmax(); out[~A] = tab[~A, bi]; print(f"   {nm} half {h} chose beta {grid[bi]}")
        print(f"   {nm} in-sample curve:", " ".join(f"{b:+.1f}:{v - D.base.mean():+.5f}" for b, v in zip(grid, tab.mean(0))))
        D.report(out, f"V {regime} {nm} nested")
