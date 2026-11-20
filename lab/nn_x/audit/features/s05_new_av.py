"""Adversarial validation + label relation + support for the NEW features in out/feats (val/test; trRL/trProd when built).
Same metrics as s01 (candidate AUC pairs, feed-level aggregate AUCs, within-feed label AUC and slope per population,
private-outside-val-support shares). Output: out/new_feature_table.csv + out/s05.log."""
import sys, time
sys.path.insert(0, "<repo>/nn_x/audit/features")
from lib_av import *
HERE = Path(__file__).resolve().parent; NEWF = HERE / "out/feats"
LOG = open(HERE / "out/s05.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


fv, Mv, offv = meta("val"); ft, Mt, offt = meta("test")
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
dayv = (fv["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
fidt = np.repeat(np.arange(len(Mt)), Mt); pub_feed = ft["timestamp"].to_numpy() < BOUND; pub_row = pub_feed[fidt]
ftr, Mtr, offtr, subs = train_subsets()
ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
TR = {}
for k in ("trRL", "trProd"):
    if (NEWF / k).exists():
        r = rows_of(offtr, Mtr, subs[k]); TR[k] = dict(M=Mtr[subs[k]], off=local_off(Mtr[subs[k]]), y=np.asarray(ytr[r], np.float64))
val50 = np.where(Mv >= 50)[0]; VALPOP = {"valAll": np.arange(len(Mv)), "val50": val50,
                                         "valTue": np.where(dayv <= dayv.min() + 1)[0], "valThu": np.where(dayv >= dayv.min() + 2)[0]}
VALROWS = {k: rows_of(offv, Mv, v) for k, v in VALPOP.items()}
priv50_rows = rows_of(offt, Mt, np.where(~pub_feed & (Mt >= 50))[0])
cold_t = np.load(feat_file("test", "it_hist_all_luo")) == 0; cold_v = np.load(feat_file("val", "it_hist_all_luo")) == 0
names = sorted(p.stem for p in (NEWF / "val").glob("*.npy"))
rows = []
for name in names:
    xv = np.load(NEWF / "val" / f"{name}.npy").astype(np.float32); xt = np.load(NEWF / "test" / f"{name}.npy").astype(np.float32)
    xtr = {k: np.load(NEWF / k / f"{name}.npy").astype(np.float32) for k in TR if (NEWF / k / f"{name}.npy").exists()}
    r = {"feature": name, "nan_val": float(np.isnan(xv).mean()), "nan_priv": float(np.isnan(xt[~pub_row]).mean()),
         "mean_val": float(np.nanmean(xv)), "sd_val": float(np.nanstd(xv)), "mean_pub": float(np.nanmean(xt[pub_row])), "mean_priv": float(np.nanmean(xt[~pub_row]))}
    r["smd_val_priv"] = (r["mean_priv"] - r["mean_val"]) / (r["sd_val"] + 1e-12)
    pairs = {"val~pub": (xv, xt[pub_row]), "val~priv": (xv, xt[~pub_row]), "pub~priv": (xt[pub_row], xt[~pub_row])}
    v50 = xv[VALROWS["val50"]]
    if "trRL" in xtr:
        pairs["trRL~val50"] = (xtr["trRL"], v50); pairs["trRL~priv50"] = (xtr["trRL"], xt[priv50_rows])
        r["nan_trRL"] = float(np.isnan(xtr["trRL"]).mean()); r["mean_trRL"] = float(np.nanmean(xtr["trRL"]))
        r["smd_trRL_val"] = (r["mean_val"] - r["mean_trRL"]) / (r["sd_val"] + 1e-12)
    if "trProd" in xtr:
        pairs["trProd~val50"] = (xtr["trProd"], v50)
    for k, (a, b) in pairs.items():
        r[f"cAUC_{k}"] = auc2(sub(a), sub(b))
    agg_v = feed_agg(xv, offv, Mv); agg_t = feed_agg(xt, offt, Mt)
    for j, an in enumerate(("mean", "nanshare", "zeroshare", "max")):
        r[f"fAUC_{an}_val~priv"] = auc2(agg_v[j], agg_t[j][~pub_feed])
        if "trRL" in xtr:
            r[f"fAUC_{an}_trRL~val50"] = auc2(feed_agg(xtr["trRL"], TR["trRL"]["off"], TR["trRL"]["M"])[j], agg_v[j][val50])
    pops = {k: (xv[VALROWS[k]], yv[VALROWS[k]], local_off(Mv[fx]), Mv[fx]) for k, fx in VALPOP.items()}
    for k in xtr:
        pops[k] = (xtr[k], TR[k]["y"], TR[k]["off"], TR[k]["M"])
    for k, (x_, y_, o_, m_) in pops.items():
        r[f"wAUC_{k}"] = feed_auc(x_.astype(np.float64), y_, o_, m_)[0]
        th, llg = clogit_slope(feed_z_nanmin(x_.astype(np.float64), o_, m_), y_, o_, m_); r[f"slope_{k}"] = th; r[f"llgain_{k}"] = llg
    xvn = xv[~np.isnan(xv)]
    if len(xvn):
        q1, q99 = np.quantile(xvn, 0.001), np.quantile(xvn, 0.999)
        xx = xt[~pub_row]; ok = ~np.isnan(xx); cc = ok & cold_t[~pub_row]
        r["out_q_priv"] = float(((xx < q1) | (xx > q99))[ok].mean()); r["out_q_cold_priv"] = float(((xx < q1) | (xx > q99))[cc].mean()) if cc.any() else np.nan
    zv = feed_z_nanmin(xv.astype(np.float64), offv, Mv); zt = feed_z_nanmin(xt.astype(np.float64), offt, Mt)
    r["zcold_val"] = float(zv[cold_v].mean()); r["zcold_priv"] = float(zt[~pub_row & cold_t].mean())
    rows.append(r)
    log(f"{name:22s} cAUC val~priv {r['cAUC_val~priv']:.3f} trRL~val50 {r.get('cAUC_trRL~val50', np.nan):.3f} trProd~val50 {r.get('cAUC_trProd~val50', np.nan):.3f} "
        f"wAUC val {r['wAUC_valAll']:.3f} trRL {r.get('wAUC_trRL', np.nan):.3f} trProd {r.get('wAUC_trProd', np.nan):.3f} slope val {r['slope_valAll']:+.3f} trProd {r.get('slope_trProd', np.nan):+.3f} out_q_priv {r.get('out_q_priv', np.nan):.4f}")
import pandas as pd
pd.DataFrame(rows).to_csv(HERE / "out/new_feature_table.csv", index=False)
log("DONE")
