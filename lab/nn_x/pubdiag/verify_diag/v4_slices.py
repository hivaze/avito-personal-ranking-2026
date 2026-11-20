"""Holdout weekend/weekday split, cold promotion (label-free) on val/public/private/holdout, cold-positive
payoff on val vs holdout, users-without-train-feeds slice, per-feed pool agreement val/public/private."""
from v_lib import *
import polars as pl

ftr, Mtr, otr = feed_meta("train")
pr = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[Mtr < 50] = 0.0
cand = np.where((pr > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
_, Mh, oh = meta("holdout")
assert (Mtr[hold] == Mh).all(), "holdout reconstruction mismatch"
print("holdout reconstruction OK; M match")
tsh = ftr["timestamp"].to_numpy()[hold]
dow = ((tsh + 10800) // 86400 + 3) % 7  # 0=Mon (1970-01-01 was Thu)
wkend = dow >= 5
print("holdout weekend share", wkend.mean().round(3))
hours = ((tsh + 10800) % 86400) / 3600
publike = (dow == 5) | ((dow == 6) & (hours < 11.53))

POOLS2 = dict(POOLS); POOLS2.update({"M2c": ["F7", "F7"]})
Hh = {k: pf("holdout", pool_final(f, "holdout")) for k, f in POOLS2.items()}
for a, b in [("M2", "M2c"), ("M2", "Cd"), ("M2", "PF7x"), ("G3", "PF7x")]:
    d = Hh[a] - Hh[b]
    df = d[wkend].mean() - d[~wkend].mean(); se = np.hypot(se_iid(d[wkend]), se_iid(d[~wkend]))
    print(f"holdout {a}-{b}: weekday {d[~wkend].mean():+.5f}+-{se_iid(d[~wkend]):.5f} weekend {d[wkend].mean():+.5f}+-{se_iid(d[wkend]):.5f} "
          f"diff {df:+.5f}+-{se:.5f} | publike {d[publike].mean():+.5f}+-{se_iid(d[publike]):.5f}")
# F7x-F7b member-root-averaged on holdout by weekend
dd = np.mean([pf("holdout", np.load(member_file(f"r6_cd_m02_r{r}", "holdout", "to"))) - pf("holdout", np.load(member_file(f"r6_cd_c0_r{r}", "holdout", "to"))) for r in range(101, 108)], 0)
print(f"holdout F7x-F7b member avg: weekday {dd[~wkend].mean():+.5f} weekend {dd[wkend].mean():+.5f} all {dd.mean():+.5f}+-{se_iid(dd):.5f}")

# ---------- cold flags
def cold_flag(split):
    if split == "holdout":
        x = np.load("<repo>/data/feats/train/it_hist_all_luo.npy", mmap_mode="r")
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(otr[hold], Mtr[hold])])
        return np.asarray(x[rows]) == 0
    return np.load(f"<repo>/data/feats/{split}/it_hist_all_luo.npy") == 0

fv, Mv, ov = meta("val"); ft_, Mt, ot = meta("test")
pub = ft_["timestamp"].to_numpy() < 1790497920
cv, ct, ch = cold_flag("val"), cold_flag("test"), cold_flag("holdout")
print("cold candidate share: val %.3f pub %.3f priv %.3f holdout %.3f" % (
    cv.mean(), ct[np.repeat(pub, Mt)].mean(), ct[np.repeat(~pub, Mt)].mean(), ch.mean()))

def cold_push(split, a, b, cold, mask=None):
    f, M, o = meta(split)
    za = z(split, pool_final(POOLS2[a], split)); zb = z(split, pool_final(POOLS2[b], split))
    dz = za - zb
    fid = np.repeat(np.arange(len(M)), M)
    nc = np.bincount(fid, weights=cold, minlength=len(M)); nw = M - nc
    sc = np.bincount(fid, weights=dz * cold, minlength=len(M)); sw = np.bincount(fid, weights=dz * (~cold), minlength=len(M))
    ok = (nc > 0) & (nw > 0)
    if mask is not None:
        ok &= mask
    return float((sc[ok] / nc[ok] - sw[ok] / nw[ok]).mean())

for a, b in [("M2", "Cd"), ("G3", "Cd"), ("PF7x", "Cd")]:
    print(f"cold push {a}-{b}: val {cold_push('val',a,b,cv):+.4f} pub {cold_push('test',a,b,ct,pub):+.4f} "
          f"priv {cold_push('test',a,b,ct,~pub):+.4f} holdout {cold_push('holdout',a,b,ch):+.4f}")

# ---------- cold-positive payoff (TO half on val; holdout = TO)
def posclass(split, cold):
    f, M, o = meta(split); y = y_of(split)
    fid = np.repeat(np.arange(len(M)), M)
    pc = np.bincount(fid, weights=y * cold, minlength=len(M)); pt = np.bincount(fid, weights=y, minlength=len(M))
    return (pc == pt) & (pt > 0), (pc == 0) & (pt > 0)
allc_v, allw_v = posclass("val", cv); allc_h, allw_h = posclass("holdout", ch)
print("all-cold-positive share: val %.3f holdout %.3f" % (allc_v.mean(), allc_h.mean()))
g = groups_val()
def to_pf(split, fams):
    return pf(split, pool_half(fams, split, "to"))
for a, b in [(POOLS["M2"], POOLS["Cd"]), (["F7b"], ["F7"]), (["F7x"], ["F7b"])]:
    dv = to_pf("val", a) - to_pf("val", b); dh = to_pf("holdout", a) - to_pf("holdout", b)
    print(f"TO {'+'.join(a)} - {'+'.join(b)}: val allcold {dv[allc_v].mean():+.5f}+-{se_iid(dv[allc_v]):.5f} allwarm {dv[allw_v].mean():+.5f}+-{se_iid(dv[allw_v]):.5f} | "
          f"holdout allcold {dh[allc_h].mean():+.5f}+-{se_iid(dh[allc_h]):.5f} allwarm {dh[allw_h].mean():+.5f}+-{se_iid(dh[allw_h]):.5f}")

# ---------- users without train feeds (val / public / private), M2-PF7x on val by slice
tr_users = set(ftr["user_id"].unique().to_list())
uv = fv["user_id"].to_numpy(); ut = ft_["user_id"].to_numpy()
nov = ~np.isin(uv, list(tr_users)); nott = ~np.isin(ut, list(tr_users))
print("users-without-train-feeds share: val %.3f pub %.3f priv %.3f" % (nov.mean(), nott[pub].mean(), nott[~pub].mean()))
Vf = {k: pf("val", pool_final(POOLS[k], "val")) for k in ["PF7x", "M2", "Cd"]}
d = Vf["M2"] - Vf["PF7x"]
print(f"M2-PF7x val: no-train-users {d[nov].mean():+.5f}+-{se_clu(d[nov], g[nov]):.5f}  others {d[~nov].mean():+.5f}+-{se_clu(d[~nov], g[~nov]):.5f}  allcold-pos {d[allc_v].mean():+.5f}+-{se_iid(d[allc_v]):.5f}")
# post-stratified M2-PF7x to public / private share of no-train users
for nm_, m in [("pub", pub), ("priv", ~pub)]:
    s = nott[m].mean()
    print(f"  poststrat({nm_}) by no-train-user share {s:.3f}: {s*d[nov].mean()+(1-s)*d[~nov].mean():+.5f}")

# ---------- per-feed Pearson of final scores (M2 vs Cd) val / pub / priv
def pfcorr(split, a, b, mask=None):
    f, M, o = meta(split)
    za = z(split, pool_final(POOLS[a], split)); zb = z(split, pool_final(POOLS[b], split))
    fid = np.repeat(np.arange(len(M)), M)
    c = np.bincount(fid, weights=za * zb, minlength=len(M)) / M
    ok = M > 1
    if mask is not None:
        ok &= mask
    return float(c[ok].mean())
for a, b in [("M2", "Cd"), ("G3", "Cd"), ("M2", "PF7x")]:
    print(f"per-feed corr {a}~{b}: val {pfcorr('val',a,b):.5f} pub {pfcorr('test',a,b,pub):.5f} priv {pfcorr('test',a,b,~pub):.5f}")
