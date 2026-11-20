"""Review R7: (i) LUO zero-rate by listing age on test public/private vs val (is the train->inference skew larger on
test?); (ii) nested (user-grouped 5-fold) post-hoc offset on the G3 FINAL pool by (age bucket x LUO==0) cells, fitted by
coordinate search on 4 folds, evaluated on the 5th: the free-ceiling of fixing the skew at inference."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"): os.environ[k] = "12"
import sys; sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/feats_v2/life_")
from common import *
import post_time as ptm
from sklearn.model_selection import GroupKFold
P = Path("<repo>/preds"); knots = np.load(FEATS / "life_f4_age_knots.npy")
bins = [0, 6, 24, 72, 168, 720, 2160, 1e9]; lab = ['<6h', '6-24h', '1-3d', '3-7d', '7-30d', '30-90d', '>90d']
ab = lambda s: np.digitize(np.nan_to_num(ptm.age_hours(np.load(FLAT / f"{s}/cand_f4.npy"), knots), nan=1e8), bins) - 1
fv, Mv, offv = feed_meta("val"); ft, Mt, offt = feed_meta("test")
bv, bt = ab("val"), ab("test"); lv, lt = np.load(FEATS / "val/it_hist_all_luo.npy"), np.load(FEATS / "test/it_hist_all_luo.npy")
pub = (ft["timestamp"].to_numpy() < 1790497920)[np.repeat(np.arange(len(Mt)), Mt)]
print("LUO==0 rate by age: " + " | ".join(f"{lab[i]} val {np.mean(lv[bv==i]==0):.2f} pub {np.mean(lt[pub&(bt==i)]==0):.2f} priv {np.mean(lt[~pub&(bt==i)]==0):.2f}" for i in range(7)))
fid = np.repeat(np.arange(len(Mv)), Mv); yv = np.load(FLAT / "val/cand_target.npy")
def z(s):
    s = s.astype(np.float64); n = len(Mv); mu = np.bincount(fid, s, n) / Mv
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / Mv - mu ** 2, 0)) + 1e-9; return (s - mu[fid]) / sd[fid]
cell = bv * 2 + (lv == 0)
uid = fv["user_id"].to_numpy(); grp = np.where(uid == 0, -1 - np.arange(len(uid)), uid)
GRID = np.array([-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3])
for name, base in [("G3 final", 0.5 * z(np.load(P / "ens_G3_ft_val.npy")) + 0.5 * z(np.load(P / "ens_G3_to_val.npy"))),
                   ("G3 TO", z(np.load(P / "ens_G3_to_val.npy")))]:
    b0 = ndcg_per_feed(base.astype(np.float32), yv, offv, Mv)
    oof = np.zeros(len(Mv)); offs = []
    for tr, te in GroupKFold(5).split(np.arange(len(Mv)), groups=grp):
        trm = np.zeros(len(Mv), bool); trm[tr] = True
        o = np.zeros(14)
        for sweep in range(2):
            for c in range(14):
                best = None
                for g in GRID:
                    o2 = o.copy(); o2[c] = g
                    sc = ndcg_per_feed((base + o2[cell]).astype(np.float32), yv, offv, Mv)[trm].mean()
                    if best is None or sc > best[0]: best = (sc, g)
                o[c] = best[1]
        offs.append(o)
        oof[te] = ndcg_per_feed((base + o[cell]).astype(np.float32), yv, offv, Mv)[te]
    d = oof - b0
    print(f"{name}: nested age x LUO0 offset {d.mean():+.5f} +- {d.std()/np.sqrt(len(d)):.5f}; mean offsets (age-major, [LUO>0, LUO=0]): {np.round(np.mean(offs, 0).reshape(7, 2), 2).tolist()}")
