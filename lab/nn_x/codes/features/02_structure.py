"""Where does each f vary? eta^2 decomposition (item / feed / (feed,cat) / (user,cat) / category), constancy shares,
same item across users vs within user, anonymous users (user_id 0) and empty histories (H=0).
Splits: val, test (all), train uniform 200K-feed sample (seed 0)."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *

def const_share(v, keys):
    k = keys[0].astype(np.int64)
    for a in keys[1:]:
        k = k * (int(a.max()) + 1) + a.astype(np.int64)
    o = np.lexsort((v, k)); ks = k[o]; vs = v[o]
    gstart = np.r_[True, ks[1:] != ks[:-1]]
    gid = np.cumsum(gstart) - 1
    newv = np.r_[True, (ks[1:] != ks[:-1]) | (vs[1:] != vs[:-1])]
    nlev = np.bincount(gid, newv)
    cnt = np.bincount(gid)
    multi = cnt >= 2
    return (nlev[multi] == 1).mean(), multi.sum()

for split in ["val", "test", "train"]:
    fidx = train_sample("uniform", 200_000, 0) if split == "train" else None
    d = load(split, fidx)
    feed = d["feed"]; cat = d["cat"].astype(np.int64); item = d["item"].astype(np.int64); user = d["user_id"].astype(np.int64)
    # item appearing in >=3 feeds
    _, iinv, icnt = np.unique(item, return_inverse=True, return_counts=True)
    print(f"\n===== {split}: feeds {d['n_feeds']} rows {len(feed)}; rows whose item is in >=3 feeds: {(icnt[iinv]>=3).mean():.3f}")
    print("f   | eta2 item(>=3) | eta2 feed | eta2 (feed,cat) | eta2 (user,cat) | eta2 cat | const in (feed,cat) | const in item(>=2 rows) | const in (user,item)")
    for k in FS:
        v = d[k].astype(np.float64)
        e_item, cov = eta2_multi(v, [item], 3)
        e_feed = eta2(v, feed)
        e_fc, _ = eta2_multi(v, [feed, cat])
        e_uc, _ = eta2_multi(v, [user, cat])
        e_c = eta2(v, cat)
        cs_fc, nfc = const_share(v, [feed, cat])
        cs_it, nit = const_share(v, [item])
        cs_ui, nui = const_share(v, [user, item])
        print(f"{k}  | {e_item:.3f} | {e_feed:.3f} | {e_fc:.3f} | {e_uc:.3f} | {e_c:.3f} | {cs_fc:.3f} (n={nfc}) | {cs_it:.3f} (n={nit}) | {cs_ui:.3f} (n={nui})")
    # same item: different users vs same user (train only has enough repeats)
    if split == "train":
        # pairs of appearances of the same item: compare within-user vs between-user absolute differences
        o = np.lexsort((d["timestamp"], user, item))
        it_s = item[o]; us = user[o]
        same_it = it_s[1:] == it_s[:-1]
        same_us = us[1:] == us[:-1]
        print("mean |diff| between consecutive appearances of the same item: same user (different feed) vs different users")
        for k in FS:
            v = d[k][o].astype(np.float64)
            dv = np.abs(np.diff(v))
            a = dv[same_it & same_us & (us[1:] > 0)]; b = dv[same_it & ~same_us]
            print(f"  {k}: same user {a.mean():.4f} (exact-equal {np.mean(a==0):.3f}, n={len(a)}) | different users {b.mean():.4f} (exact-equal {np.mean(b==0):.3f}, n={len(b)})")
    # anonymous users and empty histories
    u0 = user == 0; h0 = d["H"] == 0
    print(f"user_id 0: rows {u0.mean():.4f} (feeds {np.unique(feed[u0]).size}); H=0 rows {h0.mean():.4f}")
    floors = {"f1": 1/56, "f2": 0.0, "f3": 1/62, "f4": 1/64, "f5": 1/62}
    for k in FS:
        v = d[k]
        fl = floors[k]
        def desc(sel):
            x = v[sel]
            return f"mean {x.mean():.3f} sd {x.std():.3f} floor {np.mean(np.abs(x-fl)<1e-6):.3f}"
        print(f"  {k}: user0 [{desc(u0)}] | user>0 [{desc(~u0)}] | H=0 [{desc(h0)}] | user0&H>0 [{desc(u0 & ~h0)}]")
    for k in ["f1", "f3", "f2"]:
        v = d[k].astype(np.float64)
        for lab, sel in [("user0", u0), ("user>0", ~u0), ("H=0", h0)]:
            if sel.sum() < 1000: continue
            e_fc, _ = eta2_multi(v[sel], [feed[sel], cat[sel]])
            e_c = eta2(v[sel], cat[sel])
            e_f = eta2(v[sel], feed[sel])
            # does the value vary across feeds for the same category? between-feed sd of (feed,cat) means within category
            print(f"  {k} {lab}: eta2 (feed,cat) {e_fc:.3f}, eta2 feed {e_f:.3f}, eta2 category alone {e_c:.3f}")
