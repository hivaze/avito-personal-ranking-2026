"""Part 4: is the val->public shortfall of M2 within noise + selection?
 (a) winner's curse: decide4-style candidate pools vs PF7x; under a null of equal true deltas, how large is the val
     delta of the val-best candidate? (parametric: MVN with the empirical covariance of per-feed deltas / n)
 (b) cluster overdispersion: are per-feed SEs too small? chi2/df of (day x hour) cell means on val and on holdout.
 (c) holdout train-only analogues of the candidate pools."""
from shiftlib import *  # noqa: F401,F403

fv, Mv, ov = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
R = dict(np.load(OUT / "p2_val_perfeed.npz"))
CAND = {"Pall": None, "G3": None, "M2": None,
        "G70": FAM["F10"] + FAM["F7x"],
        "PF7x+F7x70": POOLS["PF7x"] + FAM["F7x70"],
        "F7x": None}
for k, mem in CAND.items():
    if mem is not None:
        sv, _, _ = split_pool(mem, "val", Mv)
        R[k] = ndcg_per_feed(sv.astype(np.float32), y, ov, Mv)
names = list(CAND)
D = np.stack([R[k] - R["PF7x"] for k in names], 1)
mu = D.mean(0); C = np.cov(D.T) / len(D); se = np.sqrt(np.diag(C))
print("val deltas vs PF7x: " + " | ".join(f"{k} {m:+.5f}±{s:.5f}" for k, m, s in zip(names, mu, se)))
print("correlation of the candidate deltas:\n", np.round(np.corrcoef(D.T), 2))
rng = np.random.default_rng(0)
Z = rng.multivariate_normal(np.zeros(len(names)), C, size=200000)
for label, cols in [("all 6 (incl. F7x alone)", list(range(len(names)))),
                    ("decide4 pools (Pall,G3,M2,G70,PF7x+F7x70)", [0, 1, 2, 3, 4])]:
    zz = Z[:, cols]
    mx = zz.max(1)
    print(f"[{label}] null E[max val delta] {mx.mean():+.5f}, P(max >= observed M2 {mu[2]:+.5f}) {np.mean(mx >= mu[2]):.4f}")
# conditional: truth = observed means shrunk to 0 by factor s; bias of the val-best's estimate
for s in (0.0, 0.5, 1.0):
    true = s * mu
    sim = true + Z
    pick = sim[:, :5].argmax(1)
    est = sim[np.arange(len(sim)), pick]; tr = true[pick]
    print(f"truth = {s:.1f} x observed: E[val(picked) - true(picked)] = {np.mean(est - tr):+.5f}; picked M2 {np.mean(pick == 2):.2f}")

# (b) overdispersion of cluster means
print("\n== cluster overdispersion (chi2/df of cell means around the overall mean, per-feed SE within cells) ==")


def overdisp(d, cell, label):
    u = np.unique(cell); m0 = d.mean(); chi = 0; df = 0
    for c in u:
        x = d[cell == c]
        if len(x) < 30:
            continue
        chi += (x.mean() - m0) ** 2 / (x.var(ddof=1) / len(x)); df += 1
    df -= 1
    print(f"{label:40s} cells {df + 1:4d} chi2/df {chi / df:.2f}  (p~{1 - __import__('scipy.stats', fromlist=['chi2']).chi2.cdf(chi, df):.3f})")
    return chi / df


ts = fv["timestamp"].to_numpy().astype(np.int64)
cell_dh = msk_day(ts) * 100 + msk_hour(ts)
for a, b in [("M2", "PF7x"), ("G3", "PF7x"), ("M2", "Cd"), ("F7x", "F7b"), ("F7x", "F7")]:
    overdisp(R[a] - R[b], cell_dh, f"val {a}-{b} by day x hour")
H = dict(np.load(OUT / "p1_holdout_perfeed.npz"))
hcell = msk_day(H["ts"]) * 100 + msk_hour(H["ts"])
for a, b in [("M2", "PF7x"), ("G3", "PF7x"), ("M2", "Cd"), ("F7x", "F7b"), ("F7x", "F7")]:
    overdisp(H[a] - H[b], hcell, f"holdout {a}-{b} by day x hour")
    overdisp(H[a] - H[b], msk_day(H["ts"]), f"holdout {a}-{b} by day")

# (c) holdout analogues (train-only pools)
print("\n== holdout (train-only) deltas vs PF7x of the same candidate pools ==")
hold, fh, Mh, offh = holdout_index(); yh = np.load(P / "holdout/holdout_y.npy")
for k in names:
    mem = CAND[k] if CAND[k] is not None else POOLS[k] if k in POOLS else POOLS["PF7x"] + FAM["F7x70"]
    if k == "Pall":
        continue
    s = holdout_pool(mem, Mh)
    d = ndcg_per_feed(s.astype(np.float32), yh, offh, Mh) - H["PF7x"]
    print(f"{k:12s} holdout {fmt(*mse(d))} | val {fmt(*mse(R[k] - R['PF7x']))} | val to-half n/a")
