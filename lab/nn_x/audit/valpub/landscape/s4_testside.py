"""s4: label-free TEST-side diagnostics. (a) each pool's disagreement with G3 (1 - within-feed corr, and top-1 disagreement) by
feed type on val / public / private, and the private-vs-val ratio within the same feed type (a pool whose disagreement with G3
grows on private feeds of the SAME type is extrapolating differently there); (b) by test day (Sat/Sun/Mon/Tue) and MSK hour;
(c) cold-candidate promotion: the share of cold (it_hist_all_luo == 0) candidates among each pool's top-5 on val / public /
private vs the label-free expectation of the cold-positive share (val relation P(pos cold | feed cold share bin) applied to
the test composition); (d) old-generation files: are the new-vs-old differences larger on private than on public?"""
from lib import *  # noqa

f, M, off = meta("val"); y = val_labels()
ft, Mt, ot = meta("test"); pubm = public_mask()
dv = descriptors("val"); dt = descriptors("test")
Sv = slicings(dv); St = slicings(dt)
pub_t, priv_t = dt["public"], ~dt["public"]
names = [n for n in SCORED + UNSCORED if n != "G3"]
fid_v = np.repeat(np.arange(len(M)), M); fid_t = np.repeat(np.arange(len(Mt)), Mt)


def zc(n, split):
    p = OUT / f"z_{n}_{split}.npy"
    if p.exists():
        return np.load(p)
    f_, M_, o_ = meta(split)
    z = zf(o_, M_, pool_scores(n, split)).astype(np.float32); np.save(p, z); return z


Zv = {n: zc(n, "val") for n in names + ["G3"]}; Zt = {n: zc(n, "test") for n in names + ["G3"] + OLD}
corr_v = {n: np.bincount(fid_v, Zv[n] * Zv["G3"], len(M)) / M for n in names}
corr_t = {n: np.bincount(fid_t, Zt[n] * Zt["G3"], len(Mt)) / Mt for n in names + OLD}
top_v = {n: top_item(pool_scores(n, "val"), M, off) for n in names + ["G3"]}
top_t = {n: top_item(pool_scores(n, "test"), Mt, ot) for n in names + ["G3"] + OLD}

print("=" * 150)
print("(4a) DISAGREEMENT WITH G3 by feed type: (1 - mean within-feed corr) x 1e3 on val | public | private, ratio priv/val;")
print("     top-1 disagreement rate val | public | private. Flag '*' = priv/val ratio outside [0.8, 1.25] with n_priv >= 300.")
print("=" * 150)
rows = []
for var in ["cold_share", "luo_cold", "hour", "M", "user", "weekend", "H"]:
    labs = sorted(set(Sv[var]) | set(St[var]))
    print(f"\n  == {var} ==  n_val n_pub n_priv | " + " | ".join(f"{n:>22s}" for n in names))
    for lab in labs:
        mv = Sv[var] == lab; mp = (St[var] == lab) & pub_t; mq = (St[var] == lab) & priv_t
        cells = []
        for n in names:
            a = (1 - corr_v[n][mv]).mean() * 1e3 if mv.sum() else np.nan
            b = (1 - corr_t[n][mp]).mean() * 1e3 if mp.sum() else np.nan
            c = (1 - corr_t[n][mq]).mean() * 1e3 if mq.sum() else np.nan
            t1v = (top_v[n] != top_v["G3"])[mv].mean() if mv.sum() else np.nan
            t1q = (top_t[n] != top_t["G3"])[mq].mean() if mq.sum() else np.nan
            ratio = c / a if (a and mv.sum() >= 100 and mq.sum() >= 300) else np.nan
            flag = "*" if (not np.isnan(ratio) and (ratio < 0.8 or ratio > 1.25)) else " "
            cells.append(f"{a:5.2f}|{b:5.2f}|{c:5.2f} r{ratio:4.2f}{flag}")
            rows.append(dict(var=var, slice=lab, pool=n, dis_val=a, dis_pub=b, dis_priv=c, ratio=ratio, top1dis_val=t1v, top1dis_priv=t1q,
                             n_val=int(mv.sum()), n_pub=int(mp.sum()), n_priv=int(mq.sum())))
        print(f"    {lab:12s} {mv.sum():5d} {mp.sum():5d} {mq.sum():6d} | " + " | ".join(f"{c:>22s}" for c in cells))
pl.DataFrame(rows).write_csv(OUT / "s4_disagreement.csv")

print("\n(4a') overall: disagreement with G3 x1e3 val | public | private, and top-1 disagreement val | public | private")
for n in names + OLD:
    a = (1 - corr_v[n]).mean() * 1e3 if n in corr_v else np.nan
    b = (1 - corr_t[n][pub_t]).mean() * 1e3; c = (1 - corr_t[n][priv_t]).mean() * 1e3
    tv = (top_v[n] != top_v["G3"]).mean() if n in top_v else np.nan
    tp = (top_t[n] != top_t["G3"])[pub_t].mean(); tq = (top_t[n] != top_t["G3"])[priv_t].mean()
    print(f"  {n:8s} {a:6.2f} | {b:6.2f} | {c:6.2f}   top1 {tv:.3f} | {tp:.3f} | {tq:.3f}")

# ---------------------------------------------------------------- (4b) by test day and hour
print("\n" + "=" * 150)
print("(4b) disagreement with G3 x1e3 by TEST DAY (Sat 26 / Sun 27 / Mon 28 / Tue 29, MSK) and by hour bucket within private;")
print("     and the same on val by day (Tue 22 / Thu 24). A pool that drifts away from G3 with the date is the one whose inputs go stale")
print("     differently (label-free; direction unknown).")
print("=" * 150)
dayt = shiftlib.msk_day(dt["ts"]); dayv = shiftlib.msk_day(dv["ts"])
dnames = {d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][(d + 3) % 7] for d in np.unique(np.concatenate([dayt, dayv]))}
print(f"  {'pool':8s} " + " ".join(f"{'val ' + dnames[d]:>9s}" for d in np.unique(dayv) if (dayv == d).sum() > 100) + " | " +
      " ".join(f"{'test ' + dnames[d]:>9s}" for d in np.unique(dayt) if (dayt == d).sum() > 100))
for n in names + OLD:
    cv = [f"{(1 - corr_v[n][dayv == d]).mean() * 1e3:9.2f}" if n in corr_v else f"{'n/a':>9s}" for d in np.unique(dayv) if (dayv == d).sum() > 100]
    ct = [f"{(1 - corr_t[n][dayt == d]).mean() * 1e3:9.2f}" for d in np.unique(dayt) if (dayt == d).sum() > 100]
    print(f"  {n:8s} " + " ".join(cv) + " | " + " ".join(ct))

# ---------------------------------------------------------------- (4c) cold promotion
print("\n" + "=" * 150)
print("(4c) COLD-CANDIDATE PROMOTION (it_hist_all_luo == 0). Per pool: share of cold candidates among the top-5 of each feed, mean over")
print("     feeds, on val | public | private; 'cand' = cold share of all candidates; 'exp pos' = label-free expectation of the share of")
print("     feeds whose top positive is cold (val relation P(best positive cold | cand cold-share bin) applied to the test composition).")
print("     A pool whose top-5 cold share rises less than the expectation from val to private is under-promoting cold items on private.")
print("=" * 150)
cold_v = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy") == 0
cold_t = np.load(ROOT / "data/feats/test/it_hist_all_luo.npy") == 0
cs_v = np.bincount(fid_v, cold_v, len(M)) / M; cs_t = np.bincount(fid_t, cold_t, len(Mt)) / Mt
# val relation: best positive cold?
g3pos_idx = None
order = np.lexsort((y, -pool_scores("G3", "val").astype(np.float64), fid_v))
rank = np.arange(len(order)) - np.repeat(off, M)
first_pos_cold = np.full(len(M), np.nan)
pos = y[order] > 0
idx = np.where(pos)[0]; fids = fid_v[order][idx]; uniq, firstidx = np.unique(fids, return_index=True)
first_pos_cold[uniq] = cold_v[order][idx[firstidx]]
edges = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
bv = np.searchsorted(edges, cs_v); bt = np.searchsorted(edges, cs_t)
haspos = ~np.isnan(first_pos_cold)
p_cold = np.array([first_pos_cold[haspos & (bv == b)].mean() if (haspos & (bv == b)).sum() > 30 else np.nan for b in range(len(edges) + 1)])
print("  P(best positive is cold | luo cold-share bin) on val:", np.round(p_cold, 3), " bins <.2,.2-.35,.35-.5,.5-.65,.65-.8,>=.8")
exp_v = np.nanmean(p_cold[bv[haspos]]); exp_p = np.nanmean(p_cold[bt[pub_t]]); exp_q = np.nanmean(p_cold[bt[priv_t]])
print(f"  expected cold-positive share: val {exp_v:.3f} (actual {first_pos_cold[haspos].mean():.3f}) | public {exp_p:.3f} | private {exp_q:.3f}")
print(f"  cold share of candidates: val {cs_v.mean():.3f} | public {cs_t[pub_t].mean():.3f} | private {cs_t[priv_t].mean():.3f}")


def top5_cold(s, M_, o_, cold):
    fid = np.repeat(np.arange(len(M_)), M_)
    order = np.lexsort((-np.asarray(s, np.float64), fid)); rank = np.arange(len(order)) - np.repeat(o_, M_)
    sel = rank < 5
    return np.bincount(fid[order][sel], cold[order][sel], len(M_)) / np.minimum(M_, 5)


print(f"\n  {'pool':8s} {'top5 cold val':>13s} {'public':>8s} {'private':>8s} | {'priv-val':>8s} {'(exp pos priv-val)':>18s} | {'top1 cold val':>13s} {'public':>8s} {'private':>8s}")
for n in ["G3"] + names + OLD:
    if n in OLD:
        tv = np.nan
    else:
        tv = top5_cold(pool_scores(n, "val"), M, off, cold_v).mean()
    tt = top5_cold(pool_scores(n, "test"), Mt, ot, cold_t)
    t1v = cold_v[off + top_v[n]].mean() if n in top_v else np.nan
    t1t = cold_t[ot + top_t[n]]
    print(f"  {n:8s} {tv:13.3f} {tt[pub_t].mean():8.3f} {tt[priv_t].mean():8.3f} | {tt[priv_t].mean() - tv:+8.3f} {exp_q - exp_v:+18.3f} | {t1v:13.3f} {t1t[pub_t].mean():8.3f} {t1t[priv_t].mean():8.3f}")

# ---------------------------------------------------------------- (4d) mean z given to cold candidates relative to warm, by split
print("\n(4d) mean per-feed z of cold candidates minus warm candidates (feeds with both), val | public | private: how strongly each pool")
print("     discounts cold items, and whether that discount changes on private.")
for n in ["G3"] + names:
    out = []
    for split, Z, cold, M_, fid in [("val", Zv[n], cold_v, M, fid_v), ("test", Zt[n], cold_t, Mt, fid_t)]:
        nc = np.bincount(fid, cold, len(M_)); nw = M_ - nc
        zc_ = np.bincount(fid, Z * cold, len(M_)) / np.maximum(nc, 1); zw = np.bincount(fid, Z * (~cold), len(M_)) / np.maximum(nw, 1)
        ok = (nc > 0) & (nw > 0); gap = zc_ - zw
        if split == "val":
            out.append(f"{gap[ok].mean():+.3f}")
        else:
            out.append(f"{gap[ok & pub_t].mean():+.3f} | {gap[ok & priv_t].mean():+.3f}")
    print(f"  {n:8s} val {out[0]} | public/private {out[1]}")
