"""Export the per-slice alignment tables (from 02's per-feed arrays) to CSV: out/alignment_slices_val.csv,
out/alignment_slices_hold.csv. Columns: slice, feeds_share, ndcg, headroom_share, ce_grad_share, lambda_share,
Sm_share, Sl_share, Sm_over_Sl_rel, lambda_over_ce_rel."""
import sys, csv
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT
OUT = "<repo>/nn_x/insights/labels_loss/out"

def table(R, slices, path):
    acc, nd = R["acc"], R["nd"]
    tot = dict(head=(1 - nd).sum(), gl=acc[:, 2].sum(), lam=acc[:, 3].sum(), Sm=R["Sm"].sum(), Sl=R["Sl"].sum())
    with open(path, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["slice", "feeds_share", "ndcg", "headroom_share", "ce_grad_share", "lambda_share", "Sm_share",
                    "Sl_share", "Sm_over_Sl_rel", "lambda_over_ce_rel"])
        for k, m in slices.items():
            if m.sum() == 0: continue
            sm, sl = R["Sm"][m].sum() / tot["Sm"], R["Sl"][m].sum() / tot["Sl"]
            gl, lam = acc[m, 2].sum() / tot["gl"], acc[m, 3].sum() / tot["lam"]
            w.writerow([k, round(m.mean(), 4), round(nd[m].mean(), 5), round((1 - nd[m]).sum() / tot["head"], 4),
                        round(gl, 4), round(lam, 4), round(sm, 4), round(sl, 4), round(sm / sl, 3), round(lam / gl, 3)])

MB = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 301]
R = dict(np.load(f"{OUT}/02_perfeed_val.npz"))
f, M, off = feed_meta("val")
K = R["acc"][:, 11]; best = R["acc"][:, 10] / M
y = np.asarray(np.load(FLAT / "val/cand_target.npy")).astype(np.int64)
cold = np.asarray(np.load("<repo>/nn_x/r6/feat_cache/val/it_hist_all_luo.npy")) == 0
fid = np.repeat(np.arange(len(M)), M)
nc = np.bincount(fid, weights=(cold & (y == 1)).astype(float), minlength=len(M))
sl = {f"M[{a},{b})": (M >= a) & (M < b) for a, b in zip(MB[:-1], MB[1:])}
sl.update({"K=1": K == 1, "K=2": K == 2, "K=3": K == 3, "K>=4": K >= 4,
           "allpos_cold": nc == K, "somecold": (nc > 0) & (nc < K), "allpos_warm": nc == 0,
           "best_pos_top10pct": best < 0.1, "best_pos_10_30pct": (best >= 0.1) & (best < 0.3),
           "best_pos_30_60pct": (best >= 0.3) & (best < 0.6), "best_pos_bottom40pct": best >= 0.6})
table(R, sl, f"{OUT}/alignment_slices_val.csv")
Rh = dict(np.load(f"{OUT}/02_perfeed_hold.npz"))
hm = np.load("<repo>/preds/holdout/holdout_meta.npy"); Mh = hm[0]
Kh = Rh["acc"][:, 11]
slh = {f"M[{a},{b})": (Mh >= a) & (Mh < b) for a, b in zip(MB[:-1], MB[1:])}
slh.update({"K=1": Kh == 1, "K=2": Kh == 2, "K=3": Kh == 3, "K>=4": Kh >= 4})
table(Rh, slh, f"{OUT}/alignment_slices_hold.csv")
print(open(f"{OUT}/alignment_slices_val.csv").read())
