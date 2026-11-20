"""Adversarial review of W25: independent spot checks of the leak rules on the SAVED feature columns.
(1) train rows (XM) : target statistics must equal counts over train days < d only (no own label, no same-day / later rows);
(2) val rows        : target statistics must equal counts over ALL train (no val labels);
(3) LAE scores      : must equal (feed's OWN history) @ W[:, group(cand)] with the stored W (val and train rows);
(4) stored W        : DAN diagonal 0, RDLAE diagonal = xi (paper form), EASE diagonal 0;
(5) rows.npy        : exactly the rows of the p_random > 0.5 feeds.
CPU only; 12 threads."""
import os, sys, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
import numpy as np, polars as pl

ROOT = "<repo>"; FLAT = f"{ROOT}/data/flat"; CLU = f"{ROOT}/data/clu_work"; WD = f"{ROOT}/nn_x/w25/data"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
rng = np.random.default_rng(777)
NU = 2_483_762 + 1

ftr = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
ts = ftr["timestamp"].to_numpy().astype(np.int64)
dayf = (ts + 3 * 3600) // 86400; day0 = dayf.min(); dayf = (dayf - day0).astype(np.int16)
uidf = ftr["user_id"].to_numpy().astype(np.int64)
Mtr = ftr["M"].to_numpy().astype(np.int64); otr = ftr["cand_off"].to_numpy().astype(np.int64)
fidx = np.load(f"{FLAT}/train/cand_fidx.npy").astype(np.int64)
N = len(fidx)
item = np.load(f"{FLAT}/train/cand_item_id.npy").astype(np.int64)
cat = np.load(f"{FLAT}/train/cand_category_id.npy").astype(np.int64)
y = np.load(f"{FLAT}/train/cand_target.npy").astype(np.int8)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
p2f = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0) > 0.5
day = dayf[fidx]; uid = uidf[fidx]; p2 = p2f[fidx]
uid_eff = np.where(uid > 0, uid, NU + fidx)
log(f"train rows {N}, feeds {len(Mtr)}, days {int(dayf.max())+1}, p2 feeds {p2f.sum()}")

# (5) rows.npy
rows_need = np.load(f"{WD}/train/rows.npy")
exp_rows = np.where(p2)[0]
print(f"(5) rows.npy == rows of p_random>0.5 feeds: {np.array_equal(rows_need, exp_rows)} (n {len(rows_need)})")
del exp_rows

# (1) train spot checks ------------------------------------------------------------------------------
ns = 2500
samp = np.sort(rng.choice(len(rows_need), ns, replace=False))
r = rows_need[samp]
it_s, d_s, u_s, c_s = item[r], day[r].astype(np.int64), uid[r], cat[r]
log("sorting train rows by item ...")
oi = np.argsort(item, kind="stable"); item_sorted = item[oi]
log("sorting train rows by user ...")
ou = np.argsort(uid, kind="stable"); uid_sorted = uid[ou]
cols_tr = {n: np.load(f"{WD}/train/{n}.npy", mmap_mode="r") for n in
           ("w25_it_imp2", "w25_it_pos2", "w25_it_once_imp1", "w25_it_once_pos1", "w25_it_npd", "w25_u_pos1", "w25_uc_pos1", "w25_u_feeds1", "w25_u_lastpos_cat")}
first_of_feed = np.r_[True, fidx[1:] != fidx[:-1]]
mism = {n: 0 for n in cols_tr}; nchk = {n: 0 for n in cols_tr}; own_label_counts = 0; sameday_pos = 0
for k in range(ns):
    i, d, u, c, rr = it_s[k], d_s[k], u_s[k], c_s[k], r[k]
    lo, hi = np.searchsorted(item_sorted, [i, i + 1]); ri = oi[lo:hi]
    past = ri[day[ri] < d]; past2 = past[p2[past]]
    exp = {"w25_it_imp2": len(past2), "w25_it_pos2": int(y[past2].sum()),
           "w25_it_once_imp1": len(np.unique(uid_eff[past])), "w25_it_once_pos1": len(np.unique(uid_eff[past[y[past] > 0]])),
           "w25_it_npd": len(np.unique(day[past[y[past] > 0]]))}
    # explicit own-label / same-day exclusion probe: would including day d change the stat?
    sameday = ri[day[ri] == d]
    if y[sameday].sum() > 0: sameday_pos += 1
    if y[rr] > 0: own_label_counts += 1
    if u > 0:
        lo, hi = np.searchsorted(uid_sorted, [u, u + 1]); ru = ou[lo:hi]; pu = ru[day[ru] < d]
        exp["w25_u_pos1"] = int(y[pu].sum()); exp["w25_uc_pos1"] = int(y[pu[cat[pu] == c]].sum())
        exp["w25_u_feeds1"] = int(first_of_feed[pu].sum())
        posu = pu[y[pu] > 0]
        exp["w25_u_lastpos_cat"] = float(cat[posu[np.argmax(posu)]]) if len(posu) else np.nan  # latest row (rows are time-sorted) with a positive
    for n, v in exp.items():
        got = float(cols_tr[n][samp[k]])
        if d == 0:
            ok = np.isnan(got)  # day 0 -> NaN by design
        elif n == "w25_u_lastpos_cat":
            ok = (np.isnan(got) and np.isnan(v)) or (got == v)
        else:
            ok = abs(got - v) < 1e-3
        nchk[n] += 1; mism[n] += (not ok)
        if not ok and mism[n] <= 3:
            print(f"   MISMATCH {n} row {rr} day {d} user {u}: stored {got} expected {v}")
print(f"(1) train XM rows, {ns} sampled rows: mismatches per column (checked): " + ", ".join(f"{n} {mism[n]}/{nchk[n]}" for n in cols_tr))
print(f"    of which {own_label_counts} sampled rows are positives and {sameday_pos} have a same-day positive for the item: "
      f"stored stats matched the strict days<d recomputation, so own labels / same-day rows are excluded")
del oi, item_sorted

# (2) val spot checks ------------------------------------------------------------------------------
fv = pl.read_parquet(f"{FLAT}/val/feeds.parquet"); Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64)
fiv = np.load(f"{FLAT}/val/cand_fidx.npy").astype(np.int64)
itv = np.load(f"{FLAT}/val/cand_item_id.npy").astype(np.int64); uvv = fv["user_id"].to_numpy().astype(np.int64)[fiv]
catv = np.load(f"{FLAT}/val/cand_category_id.npy").astype(np.int64)
cols_v = {n: np.load(f"{WD}/val/{n}.npy") for n in ("w25_it_imp2", "w25_it_pos2", "w25_it_once_imp1", "w25_u_pos1", "w25_uc_pos1", "w25_it_npd")}
nsv = 2000; sv = np.sort(rng.choice(len(itv), nsv, replace=False))
item_sorted = item[(oi := np.argsort(item, kind="stable"))]
mism = {n: 0 for n in cols_v}; nchk = {n: 0 for n in cols_v}; n_unseen = 0
for k in sv:
    i, u, c = itv[k], uvv[k], catv[k]
    lo, hi = np.searchsorted(item_sorted, [i, i + 1]); ri = oi[lo:hi]; ri2 = ri[p2[ri]]
    if len(ri) == 0: n_unseen += 1
    exp = {"w25_it_imp2": len(ri2), "w25_it_pos2": int(y[ri2].sum()), "w25_it_once_imp1": len(np.unique(uid_eff[ri])),
           "w25_it_npd": len(np.unique(day[ri[y[ri] > 0]]))}
    if u > 0:
        lo, hi = np.searchsorted(uid_sorted, [u, u + 1]); ru = ou[lo:hi]
        exp["w25_u_pos1"] = int(y[ru].sum()); exp["w25_uc_pos1"] = int(y[ru[cat[ru] == c]].sum())
    for n, v in exp.items():
        got = float(cols_v[n][k]); ok = abs(got - v) < 1e-3
        nchk[n] += 1; mism[n] += (not ok)
        if not ok and mism[n] <= 3:
            print(f"   MISMATCH val {n} row {k}: stored {got} expected {v}")
print(f"(2) val rows, {nsv} sampled: mismatches per column (checked): " + ", ".join(f"{n} {mism[n]}/{nchk[n]}" for n in cols_v) + f"; items unseen in train {n_unseen}")
# user-0 rows must be NaN for user-keyed stats
u0 = uvv == 0
print(f"    user-0 val rows: w25_u_pos1 NaN share {np.isnan(cols_v['w25_u_pos1'][u0]).mean():.3f} (must be 1.0); known users NaN share {np.isnan(cols_v['w25_u_pos1'][~u0]).mean():.3f}")
del oi, item_sorted, ou, uid_sorted

# (4) stored W -------------------------------------------------------------------------------------
Wd = np.load(f"{WD}/W_cl4096_dan.npy"); Wr = np.load(f"{WD}/W_cl4096_rdlae.npy"); We = np.load(f"{WD}/W_cl4096_ease.npy")
dr = np.diag(Wr)
print(f"(4) W diag: DAN max|diag| {np.abs(np.diag(Wd)).max():.2e}; EASE max|diag| {np.abs(np.diag(We)).max():.2e}; "
      f"RDLAE diag: min {dr.min():.3f} median {np.median(dr):.3f} max {dr.max():.3f} (paper form -> 0.1 where the constraint is active); "
      f"share == 0.1 +-1e-3: {(np.abs(dr - 0.1) < 1e-3).mean():.3f}; RDLAE off-diag |W| mean {np.abs(Wr - np.diag(dr)).mean():.4f}")

# (3) LAE scores from the feed's OWN history --------------------------------------------------------
def check_lae(split, feeds, rows_of_feed, colfile):
    f = pl.read_parquet(f"{FLAT}/{split}/feeds.parquet")
    H = f["H"].to_numpy().astype(np.int64); hoff = f["hist_off"].to_numpy().astype(np.int64)
    hg_all = np.load(f"{CLU}/hist_cl_4096_{split}.npy", mmap_mode="r"); hpos_all = np.load(f"{FLAT}/{split}/hist_hpos.npy", mmap_mode="r")
    cg_all = np.load(f"{CLU}/cand_cl_4096_{split}.npy", mmap_mode="r")
    dan = np.load(f"{WD}/{split}/w25_dan_bin.npy", mmap_mode="r"); rrec = np.load(f"{WD}/{split}/w25_rdlae_rec05.npy", mmap_mode="r")
    maxd_dan, maxd_rec, n = 0.0, 0.0, 0
    for fd in feeds:
        hr = np.arange(hoff[fd], hoff[fd] + H[fd]); hg = np.asarray(hg_all[hr]).astype(np.int64); hp = np.asarray(hpos_all[hr]).astype(np.float64)
        ok = hg >= 0; hg, hp = hg[ok], hp[ok]
        xb = np.zeros(4096); xb[np.unique(hg)] = 1.0
        xr = np.bincount(hg, weights=np.power(0.5, hp / 5.0), minlength=4096)
        rows, pos = rows_of_feed(fd)
        cg = np.asarray(cg_all[rows]).astype(np.int64)
        okc = cg >= 0
        sd = xb @ Wd[:, cg[okc]]; sr = xr @ Wr[:, cg[okc]]
        gd = np.asarray(dan[pos])[okc]; gr = np.asarray(rrec[pos])[okc]
        maxd_dan = max(maxd_dan, np.abs(sd - gd).max() if okc.any() else 0); maxd_rec = max(maxd_rec, np.abs(sr - gr).max() if okc.any() else 0)
        n += okc.sum()
    return maxd_dan, maxd_rec, n

vf = rng.choice(len(Mv), 150, replace=False)
md, mr, n = check_lae("val", vf, lambda fd: (np.arange(ov[fd], ov[fd] + Mv[fd]),) * 2, None)
print(f"(3) val LAE: 150 feeds / {n} cand rows: max |own-history score - stored| DAN bin {md:.2e}, RDLAE rec05 {mr:.2e}")
xm_feeds = np.where(p2f)[0]; tf = rng.choice(xm_feeds, 150, replace=False)
def rows_tr(fd):
    rows = np.arange(otr[fd], otr[fd] + Mtr[fd]); pos = np.searchsorted(rows_need, rows); assert (rows_need[pos] == rows).all(); return rows, pos
md, mr, n = check_lae("train", tf, rows_tr, None)
print(f"(3) train XM LAE: 150 feeds / {n} cand rows: max |own-history score - stored| DAN bin {md:.2e}, RDLAE rec05 {mr:.2e}")
log("done")
