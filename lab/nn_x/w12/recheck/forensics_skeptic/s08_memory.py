"""Is the TO > FT edge on 'seen' users a user-level memory of their train feed, and does KD's gain / the in-sample teacher's edge sit there?
val groups: seen in train / new / guest. holdout groups: user has another train feed (any time; the models trained on it) / none / guest."""
from hx import *
V = load_val(); y, M, off, n, uid = V["y"], V["M"], V["off"], V["n"], V["uid"]; fid = V["fid"]
Z = np.load(OUT / "bank_val.npz"); Xb = Z["X"]; nb_ = [str(s) for s in Z["names"]]; uf = Xb[off, nb_.index("u_feeds")]
seen = (uid != 0) & (uf > 0); new = (uid != 0) & (uf == 0); guest = uid == 0
Hd = load_hold(); yh, Mh, oh = Hd["y"], Hd["M"], Hd["off"]
cnt = Hd["ftrain"].group_by("user_id").len(); other = Hd["f"].join(cnt, on="user_id", how="left")["len"].to_numpy() - 1
hs = (Hd["uid"] != 0) & (other > 0); hn = (Hd["uid"] != 0) & (other == 0)
tv = lambda v: zfeed(np.load(W / f"nn_{v}_trainonly_val.npy"), off, M); th = lambda v: zfeed(np.load(W / f"holdout/nn_{v}_holdout.npy"), oh, Mh)
ev = lambda s: ndcg_per_feed(s, y, off, M); eh = lambda s: ndcg_per_feed(s, yh, oh, Mh)
def contrast(a, b, seeds=(951, 952, 953, 954)):
    A = [f"w5_e10_{a}_s{s}" for s in seeds]; B = [f"w5_e10_{b}_s{s}" for s in seeds]
    dm = np.mean([ev(tv(x)) - ev(tv(z)) for x, z in zip(A, B)], 0); dp = ev(np.mean([tv(x) for x in A], 0)) - ev(np.mean([tv(z) for z in B], 0))
    hm = np.mean([eh(th(x)) - eh(th(z)) for x, z in zip(A, B)], 0); hp = eh(np.mean([th(x) for x in A], 0)) - eh(np.mean([th(z) for z in B], 0))
    print(f"{a} - {b}")
    for nm, d in (("val member", dm), ("val 4-pool", dp)):
        print(f"   {nm:11s} all {fmt(d)} | seen {fmt(d[seen])} | new {fmt(d[new])} | guest {fmt(d[guest])} | seen-new {d[seen].mean() - d[new].mean():+.5f} +- {np.hypot(se(d[seen]), se(d[new])):.5f}")
    for nm, d in (("hold member", hm), ("hold 4-pool", hp)):
        print(f"   {nm:11s} all {fmt(d)} | user has other train feeds {fmt(d[hs])} (n {hs.sum()}) | none {fmt(d[hn])} (n {hn.sum()}) | diff {d[hs].mean() - d[hn].mean():+.5f} +- {np.hypot(se(d[hs]), se(d[hn])):.5f}", flush=True)
contrast("kd05", "base"); contrast("kd05w192", "base"); contrast("kd05w192", "kx05"); contrast("kd05w192", "kx08"); contrast("kd05w192", "kd05")
# label-based diagnostic (not a feature): is TO-FT on seen users tied to the earlier feed's clicked categories?
ucy = Xb[:, nb_.index("uc_y")]; npos = np.bincount(fid, y, n); posprev = np.bincount(fid, y * (ucy > 0), n) / np.maximum(npos, 1) > 0.5
d = ev(V["to"]) - ev(V["ft"]); print(f"W8 TO-FT on seen users: positive in a previously clicked category {fmt(d[seen & posprev])} (n {(seen & posprev).sum()}) | not {fmt(d[seen & ~posprev])} (n {(seen & ~posprev).sum()})")
d2 = ev(V["base"]) - ev(zfeed(0.5 * zfeed(np.load(P / 'ens_Q2a8s3_ft_val.npy'), off, M) + 0.5 * zfeed(np.load(P / 'ens_Q2a8s3_to_val.npy'), off, M), off, M))
print(f"W8 - C3 on seen users: positive in a previously clicked category {fmt(d2[seen & posprev])} | not {fmt(d2[seen & ~posprev])}")
