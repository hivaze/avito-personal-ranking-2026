"""a03: user "mode" from the history event mix (browsing vs contacting vs messaging).
(1) per-feed shares of event types (all history and last 32); (2) positive rate, G3 NDCG and per-feed AUC of the main
single features by mode bucket: do feature directions change with mode? (3) nested LightGBM residual over G3:
base+raw features vs + mode shares (2 seeds)."""
from common_av import *

v = meta("val")
n = v["n"]
y = np.asarray(cand("val", "target"), dtype=np.float64)
s = g3("val")
fid = np.asarray(cand("val", "fidx")).astype(np.int64)
hf = np.asarray(hist("val", "fidx")).astype(np.int64)
ht = np.asarray(hist("val", "event_type")).astype(np.int64)
hs = np.asarray(hist("val", "event_source")).astype(np.int64)
hp = np.asarray(hist("val", "hpos")).astype(np.int64)
H = v["H"].astype(np.float64)
pf = ndcg_per_feed(s, y, v["off"], v["M"])
npos = np.bincount(fid, weights=y, minlength=n)

groups = {"view": [1], "contact": [2, 3], "fav": [4], "msg": [5, 6, 8], "t79": [7, 9], "t10": [10]}
sh_all, sh_32 = {}, {}
for g, ts in groups.items():
    m = np.isin(ht, ts)
    sh_all[g] = np.bincount(hf[m], minlength=n) / np.maximum(H, 1)
    m32 = m & (hp < 32)
    sh_32[g] = np.bincount(hf[m32], minlength=n) / np.maximum(np.minimum(H, 32), 1)
feedclick = np.bincount(hf[(ht == 1) & (hs == 1)], minlength=n) / np.maximum(H, 1)
srch = np.bincount(hf[(ht == 1) & (hs == 2)], minlength=n) / np.maximum(H, 1)
log("mean shares (all history): " + ", ".join(f"{g} {sh_all[g].mean():.3f}" for g in groups) + f"; feedclick(type1,src1) {feedclick.mean():.3f}; src2 {srch.mean():.3f}")

# single features for direction checks
F = {k: feat("val", k).astype(np.float64) for k in ["f1", "f2", "f3", "f4", "f5", "cos_u", "maxsim32", "it_hist_all_luo", "seq_cv_invd20", "centrality", "cos_h10", "cat_cnt_20"]}
F["it_hist_all_luo"] = np.log1p(np.maximum(F["it_hist_all_luo"], 0))
F["seq_cv_invd20"] = np.log1p(np.maximum(F["seq_cv_invd20"], 0))
F["G3"] = s
aucs = {k: feed_auc(x, y, fid, n) for k, x in F.items()}


def bucket_report(name, b, labels):
    log(f"== {name} ==")
    hdr = "bucket      feeds  posrate  npos   M     G3ndcg(±se)   " + " ".join(f"{k:>8s}" for k in F)
    log(hdr)
    for i, lab in enumerate(labels):
        m = b == i
        if m.sum() < 50:
            continue
        rows = np.repeat(m, v["M"])
        line = f"{lab:10s} {m.sum():6d} {y[rows].mean():.4f} {npos[m].mean():.3f} {v['M'][m].mean():6.1f} {pf[m].mean():.4f}±{pf[m].std() / np.sqrt(m.sum()):.4f} "
        line += " ".join(f"{np.nanmean(aucs[k][m]):8.4f}" for k in F)
        log(line)


# contact-share buckets (feeds with H >= 10)
c = sh_all["contact"] + sh_all["msg"]
ok = H >= 10
q = np.quantile(c[ok], [0.2, 0.4, 0.6, 0.8])
b = np.where(ok, np.digitize(c, q), -1)
bucket_report("contact+msg share quintiles (H>=10)", b, ["q1(low)", "q2", "q3", "q4", "q5(high)"])
# recent-mode: last-32 contact/msg share
c32 = sh_32["contact"] + sh_32["msg"]
b = np.where(ok, np.digitize(c32, [1e-9, 0.1, 0.25, 0.5]), -1)
bucket_report("last-32 contact+msg share bins [0, (0,.1), [.1,.25), [.25,.5), >=.5]", b, ["0", "0-.1", ".1-.25", ".25-.5", ">=.5"])
# favourites share
b = np.where(ok, np.digitize(sh_all["fav"], [1e-9, 0.05, 0.15, 0.3]), -1)
bucket_report("favourite (type 4) share bins", b, ["0", "0-.05", ".05-.15", ".15-.3", ">=.3"])
# feedclick vs search share of type-1
b = np.where(ok, np.digitize(srch / np.maximum(sh_all["view"], 1e-9), [1e-9, 0.1, 0.3, 0.6]), -1)
bucket_report("search share among type-1 views (src2 / type1)", b, ["0", "0-.1", ".1-.3", ".3-.6", ">=.6"])
# H buckets for reference
b = np.digitize(H, [1, 10, 50, 150, 256])
bucket_report("history length", b, ["H=0", "1-9", "10-49", "50-149", "150-255", "256"])

log("== nested LightGBM residual over G3: base+raw vs +mode shares ==")
fold = make_folds(v["groups"], 5, 0)
zb = feed_z(s, fid, n)
logM = np.log(v["M"])[fid]
logH = np.log1p(H)[fid]
catv = np.asarray(cand("val", "category_id")).astype(np.float32)
Xr = np.column_stack([zb, logM, logH] + [F[k] for k in ["f1", "f2", "f3", "f4", "f5", "cos_u", "maxsim32", "it_hist_all_luo", "seq_cv_invd20", "centrality", "cos_h10", "cat_cnt_20"]] + [catv]).astype(np.float32)
mode_cols = [sh_all[g] for g in groups] + [sh_32[g] for g in groups] + [feedclick, srch, (H == 0).astype(np.float64)]
Xm = np.hstack([Xr, np.column_stack([m_[fid] for m_ in mode_cols]).astype(np.float32)])
ci = [Xr.shape[1] - 1]
g3nd = ndcg_per_feed(s, y, v["off"], v["M"])
acc_r, acc_m = np.zeros(n), np.zeros(n)
for seed in [0, 1]:
    o_r = lgb_oof(Xr, y, v["off"], v["M"], fold, seed=seed, cat_idx=ci)
    o_m = lgb_oof(Xm, y, v["off"], v["M"], fold, seed=seed, cat_idx=ci)
    nr = ndcg_per_feed(o_r, y, v["off"], v["M"])
    nm = ndcg_per_feed(o_m, y, v["off"], v["M"])
    acc_r += nr / 2
    acc_m += nm / 2
    log(f"seed {seed}: G3 {g3nd.mean():.5f} base+raw {nr.mean():.5f} +mode {nm.mean():.5f} delta(mode) {fmt(nm, nr)} base+raw vs G3 {fmt(nr, g3nd)}")
log(f"2-seed mean: +mode vs base+raw {fmt(acc_m, acc_r)}; per slice: H>=10 {fmt(acc_m[ok], acc_r[ok])}, M<50 {fmt(acc_m[v['M'] < 50], acc_r[v['M'] < 50])}, M>=50 {fmt(acc_m[v['M'] >= 50], acc_r[v['M'] >= 50])}")
log("done")
