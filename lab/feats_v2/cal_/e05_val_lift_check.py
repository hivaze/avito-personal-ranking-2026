"""Does the train-derived (cat x hour-bucket) label lift agree with val's own (unbiased) cat x hour-bucket lift?
Val is randomly ordered -> expected positives per row = npos_feed / M. Compare weighted correlation and a
likelihood-style score: mean log-lift of the positives' category minus the within-feed expected value."""
from calcommon import *
F = feeds("val"); M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy().astype(np.int64)
fidx = cand("val", "fidx", mmap=False); y = cand("val", "target", mmap=False).astype(np.float64)
cat = cand("val", "category_id", mmap=False).astype(np.int64)
tp = time_parts(F["timestamp"].to_numpy()); hb = tp["hb"][fidx]
e = expected_pos(np.ones(len(y)), y, off, M)
Yv = np.bincount(cat * N_HB + hb, weights=y, minlength=N_CAT * N_HB).reshape(N_CAT, N_HB)
Ev = np.bincount(cat * N_HB + hb, weights=e, minlength=N_CAT * N_HB).reshape(N_CAT, N_HB)
Rc = (Yv.sum(1) + 1) / (Ev.sum(1) + 1)
Lv = np.log((Yv + 5) / (Ev * Rc[:, None] + 5))
d = np.load(f"{OUT}/demand_tables.npz")
for nm in ["lab_ctx", "lab_hb"]:
    Lt = d[nm][:, :N_HB]
    w = Yv; m = w > 20
    c = np.sum(w[m] * (Lt[m] - np.average(Lt[m], weights=w[m])) * (Lv[m] - np.average(Lv[m], weights=w[m]))) / np.sqrt(
        np.sum(w[m] * (Lt[m] - np.average(Lt[m], weights=w[m])) ** 2) * np.sum(w[m] * (Lv[m] - np.average(Lv[m], weights=w[m])) ** 2))
    # score: sum over rows of (y - e) * lift  (positive => lift ranks positives above within-feed expectation)
    x = Lt[cat, hb]
    s = (y - e) * x
    per_feed = np.bincount(np.repeat(np.arange(len(M)), M), weights=s, minlength=len(M))
    print(nm, "weighted corr train-lift vs val-lift (cells w>20):", round(float(c), 3), "cells", int(m.sum()),
          "| sum (y-e)*lift per feed: mean", f"{per_feed.mean():.2e}", "se", f"{per_feed.std() / np.sqrt(len(M)):.2e}")
# demand lift
Dc = d["Dcnt"]; s_ = Dc.sum(1) / Dc.sum(); Ex = Dc.sum(0)[None, :] * s_[:, None]; Ld = np.log((Dc + 30) / (Ex + 30))[:, :N_HB]
x = Ld[cat, hb]; s = (y - e) * x
per_feed = np.bincount(np.repeat(np.arange(len(M)), M), weights=s, minlength=len(M))
print("dem_ctx | sum (y-e)*lift per feed: mean", f"{per_feed.mean():.2e}", "se", f"{per_feed.std() / np.sqrt(len(M)):.2e}")
for nm in ["cal_lab_cat_ctx", "cal_lab_cat_hb", "cal_lab_emb_ctx", "cal_dem_cat_ctx", "cal_dem_emb_ctx", "cal_sup_cat_ctx", "cal_lab_cat_we"]:
    x = np.nan_to_num(np.load(f"{FEATS}/val/{nm}.npy").astype(np.float64))
    # remove the category main effect: demean x within category over val (so only the time-specific part is scored)
    s = (y - e) * x
    pf = np.bincount(np.repeat(np.arange(len(M)), M), weights=s, minlength=len(M))
    print(f"{nm:18s} mean (y-e)*x per feed {pf.mean():+.3e} se {pf.std() / np.sqrt(len(M)):.3e}  z={pf.mean() / (pf.std() / np.sqrt(len(M))):+.2f}")
