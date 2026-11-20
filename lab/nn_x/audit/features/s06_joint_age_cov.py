"""Joint (listing age x coverage) support check: the marginal AV misses it. Per age bucket (from f4 knots):
row share, LUO zero rate, co-vis zero rate, pre2 NaN rate, and on labelled splits P(y | cold) / P(y) and the share
of positives in the cell, for trRL (holdout), trProd, val, public, private."""
import sys
sys.path.insert(0, "<repo>/nn_x/audit/features")
from lib_av import *
HERE = Path(__file__).resolve().parent
KN = np.load(FEATS / "life_f4_age_knots.npy")


def age_days(x):
    x = x.astype(np.float64); u = x * 64; k = np.clip(np.floor(u).astype(int), 1, 62); w = u - k
    a = KN[k] * (1 - w) + KN[k + 1] * w
    return np.where(x < 63 / 64, a, 5 * 365 * 24.0) / 24.0


EDGES = [0, 1, 3, 7, 24, 76, 1e9]; LAB = ["<1d", "1-3d", "3-7d", "7-24d", "24-76d", ">76d"]
ftr, Mtr, offtr, subs = train_subsets()
fv, Mv, offv = meta("val"); ft, Mt, offt = meta("test")
pub = (ft["timestamp"].to_numpy() < BOUND)[np.repeat(np.arange(len(Mt)), Mt)]
POP = {}
for k in ("trRL", "trProd"):
    r = rows_of(offtr, Mtr, subs[k])
    g = lambda n: np.asarray(np.load(FLAT / "train" / f"{n}.npy", mmap_mode="r")[r])
    POP[k] = dict(f4=g("cand_f4"), y=g("cand_target").astype(np.float64),
                  luo=np.asarray(np.load(feat_file("train", "it_hist_all_luo"), mmap_mode="r")[r]),
                  cv=np.asarray(np.load(feat_file("train", "seq_cv_invd20"), mmap_mode="r")[r]),
                  pre2=np.asarray(np.load(feat_file("train", "pre2_fe_s"), mmap_mode="r")[r]))
POP["val"] = dict(f4=np.load(FLAT / "val/cand_f4.npy"), y=np.load(FLAT / "val/cand_target.npy").astype(np.float64),
                  luo=np.load(feat_file("val", "it_hist_all_luo")), cv=np.load(feat_file("val", "seq_cv_invd20")), pre2=np.load(feat_file("val", "pre2_fe_s")))
tf4 = np.load(FLAT / "test/cand_f4.npy"); tl = np.load(feat_file("test", "it_hist_all_luo")); tc = np.load(feat_file("test", "seq_cv_invd20")); tp = np.load(feat_file("test", "pre2_fe_s"))
POP["public"] = dict(f4=tf4[pub], y=None, luo=tl[pub], cv=tc[pub], pre2=tp[pub])
POP["private"] = dict(f4=tf4[~pub], y=None, luo=tl[~pub], cv=tc[~pub], pre2=tp[~pub])
print("age bucket | pop | row share | LUO zero | covis zero | pre2 NaN | P(y)/base | P(y|cold)/P(y) | pos share in bucket | pos share cold-in-bucket")
out = []
for name, p in POP.items():
    a = age_days(p["f4"]); b = np.digitize(a, EDGES[1:-1])
    base = p["y"].mean() if p["y"] is not None else np.nan
    for j, lab in enumerate(LAB):
        m = b == j; c = p["luo"][m] == 0
        row = dict(pop=name, age=lab, share=m.mean(), luo0=c.mean(), cv0=(p["cv"][m] == 0).mean(), pre2nan=np.isnan(p["pre2"][m]).mean())
        if p["y"] is not None:
            yy = p["y"][m]
            row.update(py_rel=yy.mean() / base, lift_cold=(yy[c].mean() / yy.mean()) if c.any() and yy.mean() > 0 else np.nan,
                       pos_share=yy.sum() / p["y"].sum(), pos_share_cold=(yy[c].sum() / p["y"].sum()))
        out.append(row)
        print(f"{lab:7s} | {name:8s} | {row['share']:.3f} | {row['luo0']:.3f} | {row['cv0']:.3f} | {row['pre2nan']:.3f} | "
              + (f"{row['py_rel']:.2f} | {row['lift_cold']:.2f} | {row['pos_share']:.3f} | {row['pos_share_cold']:.3f}" if p["y"] is not None else "- | - | - | -"))
import pandas as pd
pd.DataFrame(out).to_csv(HERE / "out/joint_age_cov.csv", index=False)
# joint cell shares: P(age bucket, cold) per population and the val-vs-private JS divergence over the 12 cells
print("\njoint cell distribution P(age, cold) (12 cells), per population:")
dist = {}
for name, p in POP.items():
    a = age_days(p["f4"]); b = np.digitize(a, EDGES[1:-1]); c = (p["luo"] == 0).astype(int)
    cell = b * 2 + c; dist[name] = np.bincount(cell, minlength=12) / len(cell)
    print(f"{name:8s}", " ".join(f"{v:.3f}" for v in dist[name]))
def js(p, q):
    m = 0.5 * (p + q); e = 1e-9
    return 0.5 * (p * np.log((p + e) / (m + e))).sum() + 0.5 * (q * np.log((q + e) / (m + e))).sum()
for a_, b_ in (("val", "public"), ("val", "private"), ("public", "private"), ("trRL", "val"), ("trProd", "val")):
    print(f"JS({a_},{b_}) = {js(dist[a_], dist[b_]):.4f}")
