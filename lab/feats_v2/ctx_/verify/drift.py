import sys, json
sys.path.insert(0, "<repo>/src")
from common import *
K = "ctx_hn_soft05,ctx_hn_rrw,ctx_hn_e1_rrw,ctx_hn_top1w,ctx_hn_margin,ctx_hn_bestrk,ctx_hs_t10,ctx_hs_d7,ctx_hs_k128,ctx_hs_e1,ctx_hs_k256,ctx_hs_e1k64,ctx_hs_k128t10,ctx_hs_mutual,ctx_nd_knn_u_dev,ctx_nd_ucent,ctx_nd_knn_h_dev,ctx_nd_knn_h,ctx_nd_max,ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_u,ctx_nd_top5u,ctx_nd_top5h".split(",")
B = 1790497920
fv, Mv, _ = feed_meta("val"); ft, Mt, _ = feed_meta("test")
tsv = np.repeat(fv["timestamp"].to_numpy(), Mv); tst = np.repeat(ft["timestamp"].to_numpy(), Mt)
dayt = (tst + 3 * 3600) // 86400; d0 = dayt.min()  # Sat26
pub = tst < B; priv = ~pub; wkend = dayt <= d0 + 1; wkday = ~wkend
priv_wkend = priv & wkend; priv_wkday = priv & wkday
def psi(ref, x, nb=10):
    r = ref[~np.isnan(ref)]
    qs = np.unique(np.quantile(r, np.linspace(0, 1, nb + 1)[1:-1]))
    def h(z):
        nan = np.isnan(z).mean(); zz = z[~np.isnan(z)]
        c = np.bincount(np.searchsorted(qs, zz, side="right"), minlength=len(qs) + 1) / len(z)
        return np.append(c, nan)
    p, q = h(ref) + 1e-4, h(x) + 1e-4
    return float(np.sum((p - q) * np.log(p / q)))
rows = []
for n in K:
    v = np.load(FEATS / "val" / f"{n}.npy"); t = np.load(FEATS / "test" / f"{n}.npy")
    r = dict(f=n, psi_val_pub=psi(v, t[pub]), psi_val_priv=psi(v, t[priv]), psi_pub_priv=psi(t[pub], t[priv]),
             psi_wkend_wkday=psi(t[wkend], t[wkday]), psi_val_testwkday=psi(v, t[wkday]), psi_privwe_privwd=psi(t[priv_wkend], t[priv_wkday]),
             nan_val=float(np.isnan(v).mean()), nan_pub=float(np.isnan(t[pub]).mean()), nan_priv=float(np.isnan(t[priv]).mean()),
             q_val=[float(x) for x in np.nanquantile(v, [.1, .5, .9])], q_pub=[float(x) for x in np.nanquantile(t[pub], [.1, .5, .9])],
             q_priv=[float(x) for x in np.nanquantile(t[priv], [.1, .5, .9])],
             med_day=[float(np.nanmedian(t[dayt == d0 + k])) for k in range(4)])
    rows.append(r)
    print(f"{n:20s} val-pub {r['psi_val_pub']:.4f} val-priv {r['psi_val_priv']:.4f} pub-priv {r['psi_pub_priv']:.4f} we-wd {r['psi_wkend_wkday']:.4f} val-testwd {r['psi_val_testwkday']:.4f} privWE-privWD {r['psi_privwe_privwd']:.4f} | nan {r['nan_val']:.3f}/{r['nan_pub']:.3f}/{r['nan_priv']:.3f} | med {r['q_val'][1]:.4g}/{r['q_pub'][1]:.4g}/{r['q_priv'][1]:.4g} | day meds {' '.join(f'{x:.4g}' for x in r['med_day'])}")
json.dump(rows, open("/tmp/scratchpad/drift.json", "w"))
# adversarial AUC with ctx features only: val vs test, test weekend vs weekday, test pub vs priv (feed-grouped 2-fold)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
Xv = load_feats("val", K); Xt = load_feats("test", K)
fiv = np.load(FLAT / "val/cand_fidx.npy"); fit = np.load(FLAT / "test/cand_fidx.npy")
def adv(XA, XB, gA, gB, tag):
    X = np.concatenate([XA, XB]); yy = np.r_[np.zeros(len(XA)), np.ones(len(XB))]; g = np.r_[gA, gB + 10**7]
    fold = (np.random.default_rng(0).random(g.max() + 1) < 0.5)[g]
    pr = np.zeros(len(yy))
    for k in [0, 1]:
        m = lgb.train(dict(objective="binary", num_leaves=31, learning_rate=0.1, min_data_in_leaf=500, feature_fraction=0.8, verbose=-1, num_threads=16), lgb.Dataset(X[fold == k], yy[fold == k]), 100)
        pr[fold != k] = m.predict(X[fold != k])
    print(f"adv AUC {tag}: {roc_auc_score(yy, pr):.4f}")
adv(Xv, Xt, fiv, fit, "val vs test")
adv(Xt[pub], Xt[priv], fit[pub], fit[priv], "test pub vs priv")
adv(Xt[wkday], Xt[wkend], fit[wkday], fit[wkend], "test weekday vs weekend")
adv(Xv, Xt[wkday], fiv, fit[wkday], "val vs test weekdays")
