"""(c) pocket 'the positive has zero leave-user-out popularity' (25% of val feeds, NDCG 0.367 vs 0.413).
Is there a LABEL-FREE feed-level indicator of the pocket under which a different blend wins?
 1 pocket reproduction + how well label-free feed indicators (cold share etc.) find it
 2 calibration of W8 on cold candidates (softmax mass vs realised positives): is there a systematic under-scoring to correct?
 3 bucketed-weight nested: per cold-share quartile a separate weight for each alternative signal (and signal x cold-candidate terms)
 4 LightGBM residual with feed-level cold indicators x candidate-level alternatives
Usage: s05_pocket_c.py val|hold"""
from hx import *
which = sys.argv[1]
D = load_val() if which == "val" else load_hold(); y, M, off, fid, n = D["y"], D["M"], D["off"], D["fid"], D["n"]; base = D["base"]
src = FEATS / ("val" if which == "val" else "train")
def ld(nm):
    x = np.load(src / f"{nm}.npy", mmap_mode="r"); x = np.asarray(x if D["rows"] is None else x[D["rows"]], np.float32); return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
luo = ld("it_hist_all_luo"); cold = (luo == 0).astype(np.float64); p0 = ndcg_per_feed(base, y, off, M); npos = np.bincount(fid, y, n)
pos_cold = np.bincount(fid, y * cold, n) / np.maximum(npos, 1) > 0.5
cs = np.bincount(fid, cold, n) / M; mlog = np.bincount(fid, np.log1p(luo), n) / M
print(f"== (c) {which}: base {p0.mean():.5f}; pocket (positive has LUO pop 0): {pos_cold.mean():.3f} of feeds, NDCG {p0[pos_cold].mean():.4f} vs {p0[~pos_cold].mean():.4f}; cold candidates {cold.mean():.3f} of rows")
def auc(s, t):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(len(s)); return (r[t].mean() - (t.sum() - 1) / 2) / (len(s) - t.sum())
ind = {"cold_share": cs, "-mean log1p(luo)": -mlog, "-M": -M.astype(float), "-H": -D["H"].astype(float)}
if which == "val":
    cnt0 = (ld("it_cand_cnt_train") == 0).astype(np.float64); ind["share not a train candidate"] = np.bincount(fid, cnt0, n) / M
for k, v in ind.items(): print(f"   feed-level indicator {k:28s} AUC for the pocket {auc(v, pos_cold):.3f}")
qs = np.quantile(cs, [0.25, 0.5, 0.75]); bq = np.searchsorted(qs, cs, side="right")
# 2 calibration on cold candidates
half = user_half(D["uid"], 0); a = _fit = None
from hx import _fit_scale
sc = _fit_scale(base, y, M, off, grid=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)); s = sc * base; mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[fid]); pr = e / np.bincount(fid, e, n)[fid]
exp_c = np.bincount(fid, pr * cold, n) * npos; obs_c = np.bincount(fid, y * cold, n)
print(f"   calibration (softmax scale {sc}): positives on cold candidates observed {obs_c.sum():.0f} vs expected {exp_c.sum():.1f} -> ratio {obs_c.sum() / exp_c.sum():.3f} (boot se {np.std([obs_c[i].sum() / exp_c[i].sum() for i in (np.random.default_rng(b).integers(0, n, n) for b in range(200))]):.3f})")
print("   cold-share quartile: feeds, pocket share, NDCG, obs/exp cold positives")
for b in range(4):
    m = bq == b; print(f"     q{b} (cold share {cs[m].min():.2f}-{cs[m].max():.2f}): {m.sum():6d}  pocket {pos_cold[m].mean():.3f}  NDCG {p0[m].mean():.4f}  obs/exp {obs_c[m].sum() / max(exp_c[m].sum(), 1e-9):.3f}")
# 3 bucketed-weight nested
alt = {"luo0 flag": zfeed(cold, off, M), "luo z (popularity)": ld("it_hist_all_luo_z").astype(np.float64)}
for nm in ("cos_u_z", "cos_hall_z", "maxsim32_z", "seq_max_all_z", "f4_z", "f5_z", "cos_het1", "pre2_fe_z", "pre2n_new_z", "it_tr_coec_z"):
    if which == "hold" and nm.startswith(("pre", "it_tr")): continue   # train-row format of these columns is not val-like (leak / parity notes in COVERAGE_MAP)
    alt[nm] = zfeed(ld(nm), off, M)
res = {}
print("   alternative signal: global-w nested | cold-share-quartile weights nested | chosen weights per quartile (fit half 1, fit half 0)")
for nm, z in list(alt.items()):
    db, dg, cw = nested_bucket(D, z, bq); res[nm] = (db, dg); print(f"     {nm:22s} {fmt(dg)} | {fmt(db)} (z {db.mean() / max(se(db), 1e-12):+.2f}) | {cw}", flush=True)
print("   signal x cold-candidate interaction terms (signal active on cold candidates only; one global weight, and per quartile):")
for nm in ("cos_u_z", "cos_hall_z", "maxsim32_z", "seq_max_all_z", "f4_z", "f5_z"):
    z = alt[nm] * cold; db, dg, cw = nested_bucket(D, z, bq); print(f"     {nm + ' x cold':22s} {fmt(dg)} | {fmt(db)} (z {db.mean() / max(se(db), 1e-12):+.2f}) | {cw}", flush=True)
# 4 LightGBM residual
ctx = ctx_cols(D); cols = dict(ctx); cols.update(cold_share=np.repeat(cs, M), mean_logluo=np.repeat(mlog, M), luo0=cold, logluo=np.log1p(luo))
for nm, z in alt.items(): cols[nm.replace(" ", "_")] = z
if which == "val": cols.update(cnt0=cnt0, cnt0_share=np.repeat(ind["share not a train candidate"], M))
X = np.stack([np.asarray(v, np.float32) for v in cols.values()], 1); d, det, _ = resid_cv(D, X, list(cols), "c_lgb (cold indicators x alternatives)")
print(f"   c_lgb inside the pocket {fmt(d[pos_cold])} / outside {fmt(d[~pos_cold])}; by cold-share quartile: " + " ".join(f"q{b} {d[bq == b].mean():+.5f}" for b in range(4)))
np.savez(OUT / f"s05_{which}.npz", d=d, pos_cold=pos_cold, cs=cs)
