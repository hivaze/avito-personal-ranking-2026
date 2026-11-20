"""(d) pocket 'users seen in train are slightly worse'. Is it a model deficiency (fixable) or feed difficulty (not)?
 1 reproduce raw + adjusted gap (OLS on M-bucket x npos x H-bucket dummies) for W8, its FT and TO branches, C3 (hard labels)
 2 difficulty probes that need no label: guest-like history, M, share of re-shown items; and a label-based one: NDCG of label-free baselines (cos_u, popularity) in the same groups
 3 fixes: seen-conditional FT/TO weight; seen-conditional weights for alternative signals (bucketed nested)
 4 holdout analogue: users with an earlier train feed (as-of) vs none
"""
from hx import *
def adj(p, g, M, npos, H, tag):
    """OLS of per-feed NDCG on group dummies with fixed effects for (M bucket x min(npos,3)) and H bucket; HC0 se"""
    mb = np.searchsorted([10, 20, 30, 50, 75, 100, 150, 200], M, side="right"); hb = np.searchsorted([1, 16, 64, 128, 256], H, side="right")
    cell = mb * 4 + np.minimum(npos, 3).astype(int); fe = np.unique(cell, return_inverse=True)[1]; fh_ = np.unique(hb, return_inverse=True)[1]
    gs = [v for v in np.unique(g) if v != 0]; X = np.zeros((len(p), len(gs) + fe.max() + 1 + fh_.max()));
    for j, v in enumerate(gs): X[:, j] = g == v
    X[np.arange(len(p)), len(gs) + fe] = 1
    for h in range(1, fh_.max() + 1): X[:, len(gs) + fe.max() + h] = fh_ == h
    XtX = np.linalg.pinv(X.T @ X); b = XtX @ X.T @ p; r = p - X @ b; V = XtX @ (X.T * r**2) @ X @ XtX
    return {v: (b[j], np.sqrt(V[j, j])) for j, v in enumerate(gs)}
V = load_val(); y, M, off, fid, n, uid = V["y"], V["M"], V["off"], V["fid"], V["n"], V["uid"]; npos = np.bincount(fid, y, n)
Z = np.load(OUT / "bank_val.npz"); Xb = Z["X"]; nb_ = [str(s) for s in Z["names"]]; uf = Xb[off, nb_.index("u_feeds")]; ugap = Xb[off, nb_.index("u_gapd")]; uy = Xb[off, nb_.index("u_y")]
grp = np.where(uid == 0, 9, np.where(uf == 0, 0, np.where(uf == 1, 1, np.where(uf <= 3, 2, 3)))); lab = {0: "new user (ref)", 1: "1 train feed", 2: "2-3 train feeds", 3: "4+ train feeds", 9: "guest (u0)"}
c3 = zfeed(0.5 * zfeed(np.load(P / "ens_Q2a8s3_ft_val.npy"), off, M) + 0.5 * zfeed(np.load(P / "ens_Q2a8s3_to_val.npy"), off, M), off, M)
ld = lambda nm: np.nan_to_num(np.load(FEATS / "val" / f"{nm}.npy").astype(np.float64))
models = {"W8": V["base"], "W8 FT branch": V["ft"], "W8 TO branch": V["to"], "C3 (hard labels)": c3, "cos_u only": ld("cos_u"), "LUO popularity only": ld("it_hist_all_luo"), "pre2_fe only": ld("pre2_fe")}
Pm = {k: ndcg_per_feed(v, y, off, M) for k, v in models.items()}
print("== (d) val: group sizes", {lab[k]: int((grp == k).sum()) for k in lab})
print("   raw NDCG by group: " + " | ".join(f"{lab[k]} {Pm['W8'][grp == k].mean():.4f}" for k in lab))
print("   adjusted gap vs new users (FE: M bucket x npos, H bucket):")
for k, p in Pm.items():
    a = adj(p, grp, M, npos, V["H"], k); print(f"     {k:20s} " + " | ".join(f"{lab[g]} {a[g][0]:+.4f} +- {a[g][1]:.4f}" for g in (1, 2, 3, 9)))
seen = (grp >= 1) & (grp <= 3); a = adj(Pm["W8"], np.where(seen, 1, np.where(grp == 9, 9, 0)), M, npos, V["H"], "W8"); print(f"   pooled 'seen in train' adjusted gap, W8: {a[1][0]:+.4f} +- {a[1][1]:.4f}  (n seen {seen.sum()})")
# paired branch differences inside the groups
for k1, k2 in (("W8 TO branch", "W8 FT branch"), ("W8", "C3 (hard labels)")):
    dd = Pm[k1] - Pm[k2]; print(f"   paired {k1} - {k2}: seen {fmt(dd[seen])} | new {fmt(dd[grp == 0])} | guest {fmt(dd[grp == 9])} | seen-minus-new {dd[seen].mean() - dd[grp == 0].mean():+.5f} +- {np.hypot(se(dd[seen]), se(dd[grp == 0])):.5f}")
# does the gap depend on the earlier feed's outcome / recency?
sub = {"seen, last train feed <= 5 d": seen & (ugap <= 5), "seen, last train feed > 5 d": seen & (ugap > 5), "seen, 1 earlier click": seen & (uy == 1), "seen, 2+ earlier clicks": seen & (uy >= 2)}
print(f"   (every seen user has >= 1 earlier click: train feeds without a positive do not exist; min u_y among seen = {uy[seen].min():.0f})")
for j, (k, m) in enumerate(sub.items()):
    gg = np.where(m, 1, np.where(seen | (grp == 9), 7, 0)); a = adj(Pm["W8"], gg, M, npos, V["H"], k); dd = Pm["W8 TO branch"] - Pm["W8 FT branch"]
    print(f"   {k:30s} n {m.sum():5d} adjusted vs new {a[1][0]:+.4f} +- {a[1][1]:.4f} | TO-FT {fmt(dd[m])}")
# label-free difficulty: history length, M, share of candidates in the user's history categories (feed_cat_share), candidates' mean cos_u
print("   label-free profile (mean): group | M | H | H==256 share | mean cos_u | cold share")
cu = np.bincount(fid, ld("cos_u"), n) / M; cs = np.bincount(fid, ld("it_hist_all_luo") == 0, n) / M
for k in lab: m = grp == k; print(f"     {lab[k]:16s} {M[m].mean():6.1f} {V['H'][m].mean():6.1f} {(V['H'][m] == 256).mean():.3f} {cu[m].mean():+.4f} {cs[m].mean():.3f}")
# 3 fixes
half = user_half(uid, 0); bk = np.where(seen, 1, np.where(grp == 9, 2, 0)); ws = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
Pw = np.stack([ndcg_per_feed(w * V["ft"] + (1 - w) * V["to"], y, off, M) for w in ws]); p0 = Pm["W8"]; d = np.zeros(n); ch = []
for a_ in (True, False):
    tr = half == a_; c = []
    for b in (0, 1, 2):
        m = bk == b; k = int(np.argmax(Pw[:, tr & m].mean(1))); c.append(ws[k]); d[(~tr) & m] = Pw[k, (~tr) & m] - p0[(~tr) & m]
    ch.append(c)
print(f"   FIX 1 group-conditional FT weight (new / seen / guest), nested: {fmt(d)} (z {d.mean() / se(d):+.2f}); halves {fmt(d[half])} / {fmt(d[~half])}; chosen w_ft {ch}; in-sample curve seen: " + " ".join(f"{w}:{(Pw[i, seen] - p0[seen]).mean():+.4f}" for i, w in enumerate(ws) if w in (0, 0.3, 0.5, 0.7, 1.0)))
print("   in-sample w_ft curve, delta vs 0.5 -- all feeds: " + " ".join(f"{w}:{(Pw[i] - p0).mean():+.5f}" for i, w in enumerate(ws)) )
for nm_, mk in (("seen", seen), ("new", grp == 0), ("guest", grp == 9)): print(f"      {nm_:6s} " + " ".join(f"{w}:{(Pw[i, mk] - p0[mk]).mean():+.5f}" for i, w in enumerate(ws)))
dd = Pm["W8 TO branch"] - Pm["W8 FT branch"]
for lo, hi in ((0, 30), (30, 100), (100, 1000)): mk = (M >= lo) & (M < hi); print(f"      TO-FT, M in [{lo},{hi}): seen {fmt(dd[mk & seen])} | new {fmt(dd[mk & (grp == 0)])}")
# same contrast for the hard-label pool C3 (is it a KD-student effect or generic?)
c3ft = zfeed(np.load(P / "ens_Q2a8s3_ft_val.npy"), off, M); c3to = zfeed(np.load(P / "ens_Q2a8s3_to_val.npy"), off, M); d3 = ndcg_per_feed(c3to, y, off, M) - ndcg_per_feed(c3ft, y, off, M)
print(f"   C3 (hard-label pool) TO-FT: all {fmt(d3)} | seen {fmt(d3[seen])} | new {fmt(d3[grp == 0])} | seen-minus-new {d3[seen].mean() - d3[grp == 0].mean():+.5f} +- {np.hypot(se(d3[seen]), se(d3[grp == 0])):.5f};  W8 TO-FT all {fmt(dd)}")
# split-seed robustness of FIX 1
for sd_ in (1, 2, 3):
    hf = user_half(uid, sd_); d_ = np.zeros(n)
    for a_ in (True, False):
        tr = hf == a_
        for b in (0, 1, 2):
            m = bk == b; k = int(np.argmax(Pw[:, tr & m].mean(1))); d_[(~tr) & m] = Pw[k, (~tr) & m] - p0[(~tr) & m]
    print(f"      FIX 1 with user-split seed {sd_}: {fmt(d_)}; halves {fmt(d_[hf])} / {fmt(d_[~hf])}")
print("   FIX 2 group-conditional weight on an alternative signal (new / seen / guest), nested: global | per group | chosen")
for nm in ("cos_u_z", "cos_hall_z", "maxsim32_z", "it_hist_all_luo_z", "pre2_fe_z", "f4_z", "f5_z", "seq_max_all_z", "it_tr_coec_z"):
    z = zfeed(ld(nm), off, M); db, dg, cw = nested_bucket(V, z, bk); print(f"     {nm:20s} {fmt(dg)} | {fmt(db)} (z {db.mean() / max(se(db), 1e-12):+.2f}) | {cw}", flush=True)
# 4 holdout analogue
Hd = load_hold(); Zh = np.load(OUT / "bank_hold_g0.npz"); nh = [str(s) for s in Zh["names"]]; ufh = Zh["X"][Hd["off"], nh.index("u_feeds")]
ph = ndcg_per_feed(Hd["base"], Hd["y"], Hd["off"], Hd["M"]); nph = np.bincount(Hd["fid"], Hd["y"], Hd["n"])
# later train feeds of the same user also exist for holdout users (the model trained on them): count all other train feeds
ftr = Hd["ftrain"]; cnt = ftr.group_by("user_id").len(); allf = Hd["f"].join(cnt, on="user_id", how="left")["len"].to_numpy() - 1
gh = np.where(Hd["uid"] == 0, 9, np.where(ufh == 0, 0, np.where(ufh == 1, 1, 2))); a = adj(ph, gh, Hd["M"], nph, Hd["H"], "hold")
print(f"== (d) holdout (W8 TO pool {ph.mean():.5f}): earlier train feeds as-of: none {int((gh == 0).sum())}, one {int((gh == 1).sum())}, 2+ {int((gh == 2).sum())}, guest {int((gh == 9).sum())}")
print("   adjusted gap vs users without an earlier feed: " + " | ".join(f"{k} {a[g][0]:+.4f} +- {a[g][1]:.4f}" for g, k in ((1, "1 earlier feed"), (2, "2+ earlier feeds"), (9, "guest"))))
g3 = np.where(Hd["uid"] == 0, 9, np.where(allf == 0, 0, np.where((ufh == 0) & (allf > 0), 1, 2))); a = adj(ph, g3, Hd["M"], nph, Hd["H"], "hold")
print(f"   by ANY other train feed (the TO models trained on them): only-later feeds n {int((g3 == 1).sum())}: {a[1][0]:+.4f} +- {a[1][1]:.4f} | has earlier feeds n {int((g3 == 2).sum())}: {a[2][0]:+.4f} +- {a[2][1]:.4f}")
