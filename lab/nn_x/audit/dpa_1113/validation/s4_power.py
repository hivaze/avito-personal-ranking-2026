"""S4: val / holdout size and power. Per-feed delta sd of every scored contrast (pool level) and of single-member
contrasts (member level), user-clustered vs iid SE, minimum detectable effects for each sample, and the reliability of
val deltas across the Tue / Thu halves. Writes out/s4_power.log.
"""
from lib import *  # noqa: F401,F403
from lib import _HOLD

LOG = open(OUT / "s4_power.log", "w")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def clustered_se(d, groups):
    d = np.asarray(d, np.float64); m = d.mean()
    u, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, d - m)
    return np.sqrt((s ** 2).sum()) / len(d)


def main():
    fv, Mv, offv = meta("val")
    uid = fv["user_id"].to_numpy().astype(np.int64)
    grp = np.where(uid > 0, uid, -1 - np.arange(len(uid)))  # user-0 feeds are singletons
    pv = {p: np.load(OUT / f"pfv_{p}.npy") for p in POOLS}
    holdout_index()
    ftr = meta("train")[0]
    huid = ftr["user_id"].to_numpy().astype(np.int64)[_HOLD["idx"]]
    hgrp = np.where(huid > 0, huid, -1 - np.arange(len(huid)))
    vu, vc = np.unique(uid[uid > 0], return_counts=True)
    pr(f"val: {len(uid)} feeds, {len(vu)} distinct non-zero users, users with >=2 val feeds {(vc >= 2).sum()} ({vc[vc >= 2].sum()} feeds)")
    hu, hc = np.unique(huid[huid > 0], return_counts=True)
    pr(f"holdout: {len(huid)} feeds, {len(hu)} distinct users, users with >=2 holdout feeds {(hc >= 2).sum()} ({hc[hc >= 2].sum()} feeds)")
    pr("\n== pool-level contrasts vs G3: per-feed delta sd, iid SE, user-clustered SE (val) | holdout ==")
    sds = []
    for p in CONTRASTS:
        d = pv[p] - pv["G3"]
        se, cse = d.std(ddof=1) / np.sqrt(len(d)), clustered_se(d, grp)
        line = f"  {p:7s} val sd {d.std():.4f} SE {se:.5f} clustered {cse:.5f} (x{cse / se:.3f})"
        fh = OUT / f"pfh_{'G3' if p == 'TOonly' else p}.npy"
        if p not in ("S2", "TOonly") and fh.exists():
            dh = np.load(fh) - np.load(OUT / "pfh_G3.npy")
            line += f" | holdout sd {dh.std():.4f} SE {dh.std() / np.sqrt(len(dh)):.5f} clustered {clustered_se(dh, hgrp):.5f}"
        pr(line); sds.append(d.std())
    # member level: single TO member vs single TO member (val), several pairs
    pr("\n== member-level contrasts (train-only val preds): per-feed delta sd ==")
    y = val_y()
    def nd(m):
        return ndcg_per_feed(np.load(P / f"nn_{m}_trainonly_val.npy").astype(np.float32), y, offv, Mv)
    pairs = [("r6_cd_m02_r101", "r6_cd_m02_r102"), ("r6_cd_m02_r101", "r6_cd_m02_p70k_r101"), ("v23_gru_pre2s_s501", "r6_cd_m02_r101"),
             ("abl_e10_pop1_s951", "r6_cd_m02_r101"), ("abl_e10_xdrop_pop_s951", "abl_e10_pop1_s951")]
    msd = []
    for a, b in pairs:
        d = nd(a) - nd(b)
        msd.append(d.std())
        pr(f"  {a} - {b}: mean {d.mean():+.5f} sd {d.std():.4f} SE {d.std() / np.sqrt(len(d)):.5f}")
    pr("\n== minimum detectable effect (two-sided alpha 0.05, power 0.8 -> 2.80 SE; the project's 2-SE bar has 50% power at 2 SE) ==")
    for lab, sd in [("pool near-copy (sd 0.037)", 0.037), ("pool family-different (sd 0.055)", 0.055),
                    (f"single member vs member (sd {np.median(msd):.3f})", float(np.median(msd)))]:
        row = f"  {lab:38s}"
        for nm, n in [("val", 14423), ("holdout", 30000), ("public", N_PUB), ("private", N_PRIV), ("val+holdout", 44423),
                      ("val+hold+rl60", 44423 + 59794)]:
            row += f" {nm} {2.8 * sd / np.sqrt(n):.5f}"
        pr(row)
    # reliability: Tue vs Thu deltas across contrasts
    tue = fv["timestamp"].to_numpy() < 1790150400  # MSK day 20719 starts at 1790150400 - 10800
    dd = (fv["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
    tue = dd <= 20719
    a = np.array([(pv[p] - pv["G3"])[tue].mean() for p in CONTRASTS]); b = np.array([(pv[p] - pv["G3"])[~tue].mean() for p in CONTRASTS])
    pr(f"\nTue-vs-Thu replication of the 8 contrast deltas: corr {np.corrcoef(a, b)[0, 1]:+.3f}; "
       + ", ".join(f"{p} {x:+.5f}/{y_:+.5f}" for p, x, y_ in zip(CONTRASTS, a, b)))
    # half-split reliability of val contrasts (random user-grouped halves), 200 splits
    rng = np.random.default_rng(0)
    D = np.stack([pv[p] - pv["G3"] for p in CONTRASTS], 1)
    ug, inv = np.unique(grp, return_inverse=True)
    cors = []
    for _ in range(200):
        h = rng.random(len(ug)) < 0.5
        m = h[inv]
        cors.append(np.corrcoef(D[m].mean(0), D[~m].mean(0))[0, 1])
    pr(f"random user-grouped half-split corr of the 8 contrast deltas: mean {np.mean(cors):+.3f}, 5-95% [{np.quantile(cors, .05):+.3f}, {np.quantile(cors, .95):+.3f}]")
    # inverse-variance val+holdout combination: does it help? (correlation of the two lenses across H6 contrasts)
    Ch = ["Cd", "Pall", "PF7x", "M2", "Q1a", "Q2a"]
    vh = np.array([(pv[p] - pv["G3"]).mean() for p in Ch]); hh = np.array([(np.load(OUT / f"pfh_{p}.npy") - np.load(OUT / "pfh_G3.npy")).mean() for p in Ch])
    pub = np.array([PUBLIC[p] - PUBLIC["G3"] for p in Ch])
    pr(f"\nH6 contrasts: corr(val, holdout) {np.corrcoef(vh, hh)[0, 1]:+.3f}; slope holdout~val {np.polyfit(vh, hh, 1)[0]:+.3f}; "
       f"corr(val, public) {np.corrcoef(vh, pub)[0, 1]:+.3f}; corr(holdout, public) {np.corrcoef(hh, pub)[0, 1]:+.3f}")
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = lam * vh + (1 - lam) * hh
        pr(f"   combo lam*val+(1-lam)*holdout lam {lam:.2f}: corr with public {np.corrcoef(c, pub)[0, 1]:+.3f}, rmse(public - combo) {np.sqrt(((pub - c) ** 2).mean()):.5f}")


if __name__ == "__main__":
    main()
