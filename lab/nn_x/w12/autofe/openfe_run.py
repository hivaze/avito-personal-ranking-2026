"""Cross-check with the OpenFE library itself (generate -> successive halving -> stage-2 importance), fitted on a
feed sample of user half A with the W8 score as init_score, then nested A -> B GBM residual test + holdout read."""
import sys, time, json
import numpy as np, pandas as pd, lightgbm as lgb
from openfe import OpenFE, transform, tree_to_formula
from lib import *
from lib import _rk

NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 12; NS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000; TOPK = 40
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
load_stats()
V = DS("val"); Hd = DS("hold")
hA = V.half == 0
cols = [b.replace(":", "_") for b in BASE]


def frame(D):
    df = pd.DataFrame({c: D.get(b) for c, b in zip(cols, BASE)})
    df["feed"] = D.fid.astype(np.int32); df["cat"] = D.cat.astype(np.int32)
    df["zb"] = D.zb
    return df


def logsm(D):
    s = D.zb.astype(np.float64)
    mx = np.maximum.reduceat(s, D.off)
    lz = np.log(np.add.reduceat(np.exp(s - np.repeat(mx, D.M)), D.off)) + mx
    return s - np.repeat(lz, D.M)


dv = frame(V); lsv = logsm(V)
rA = np.repeat(hA, V.M)
# pointwise calibration of the init score on half A: logit = a * log_softmax + b
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100).fit(lsv[rA][:, None], V.y[rA])
init_all = lr.decision_function(lsv[:, None])
rng = np.random.default_rng(0)
fa = np.where(hA)[0]; samp = np.sort(rng.choice(fa, NS, replace=False))
fm = np.zeros(V.nf, bool); fm[samp] = True
rs = np.repeat(fm, V.M)
tr_f = np.zeros(V.nf, bool); tr_f[samp[rng.random(NS) < 0.75]] = True
d_s = dv[rs].copy(); y_s = pd.DataFrame({"y": V.y[rs]}, index=d_s.index)
tri = d_s.index[np.repeat(tr_f, V.M)[rs]]; vai = d_s.index[~np.repeat(tr_f, V.M)[rs]]
print("openfe sample rows", len(d_s), "train", len(tri), "val", len(vai), flush=True)
t0 = time.time()
ofe = OpenFE()
feats = ofe.fit(data=d_s, label=y_s, task="classification", metric="binary_logloss", train_index=tri, val_index=vai,
                init_scores=pd.DataFrame({"i": init_all[rs]}, index=d_s.index), categorical_features=["feed", "cat"],
                n_data_blocks=8, min_candidate_features=1000, n_jobs=NJ, seed=1, verbose=False,
                tmp_save_path=str(CACHE / f"openfe_tmp{TAG}.feather"))
print("openfe fit", f"{time.time() - t0:.0f}s", "features returned", len(feats), flush=True)
forms = [tree_to_formula(f) for f in feats]
json.dump(forms, open(OUT / f"openfe_features{TAG}.json", "w"), indent=0)
print("top formulas:", forms[:TOPK], flush=True)
top = feats[:TOPK]
dummy = d_s.iloc[:1000].copy(); dummy["feed"] += 2_000_000  # separate groups: no contamination of feed-level stats
dv2 = dv.copy().reset_index(drop=True); dv2.index = dv2.index + 10_000_000
_, tv = transform(dummy.copy(), dv2, top, n_jobs=NJ)
dh = frame(Hd); dh["feed"] += 1_000_000; dh.index = dh.index + 20_000_000
_, th = transform(dummy.copy(), dh, top, n_jobs=NJ)
assert len(tv) == V.n and len(th) == Hd.n
newc = [c for c in tv.columns if c.startswith("autoFE_f_")]
print("new cols", len(newc), flush=True)
def softmax_obj(preds, ds):
    """exact listwise softmax CE (multi-positive) on init_score + trees; LightGBM's rank_xendcg is NOT usable as a
    residual objective here: its randomised targets pull a sharp init score back to a log(3) logit gap (-0.02 NDCG)"""
    Mg = ds.get_group().astype(np.int64); og = np.concatenate([[0], np.cumsum(Mg)[:-1]]).astype(np.int64)
    g, h, _ = softmax_gh(og, Mg, preds.astype(np.float32), ds.get_label().astype(np.float32))
    return -g.astype(np.float64), np.maximum(h, 1e-6).astype(np.float64)


P = dict(objective=softmax_obj, learning_rate=0.03, num_leaves=7, min_data_in_leaf=3000, lambda_l2=100.0,
         feature_fraction=0.5, max_bin=31, num_threads=12, verbose=-1, seed=1, min_sum_hessian_in_leaf=20.0)
base_v = V.ndcg(V.zb); base_h = Hd.ndcg(Hd.zb)
res = {}
for nm, use in (("OFE10", newc[:10]), ("OFE40", newc)):
    Xv = np.column_stack([V.zb, _rk(V.off, V.M, V.zb)] + [np.nan_to_num(tv[c].to_numpy(np.float32), nan=0, posinf=1e6, neginf=-1e6) for c in use])
    Xh = np.column_stack([Hd.zb, _rk(Hd.off, Hd.M, Hd.zb)] + [np.nan_to_num(th[c].to_numpy(np.float32), nan=0, posinf=1e6, neginf=-1e6) for c in use])

    def fit(fmask, n):
        r = np.repeat(fmask, V.M)
        return lgb.train(P, lgb.Dataset(Xv[r], label=V.y[r], group=V.M[fmask], init_score=V.zb[r].astype(np.float64)), num_boost_round=n)
    m = fit(hA & (V.inner == 0), 200); best = (0.0, 0)
    for n in (25, 50, 100, 200):
        d = (V.ndcg(V.zb + m.predict(Xv, num_iteration=n)) - base_v)[hA & (V.inner == 1)].mean()
        if d > best[0]: best = (d, n)
    n = best[1]; out = {"rounds": n, "inner_gain": best[0]}
    if n:
        m = fit(hA, n)
        d = (V.ndcg(V.zb + m.predict(Xv)) - base_v)[~hA]
        dh = Hd.ndcg(Hd.zb + m.predict(Xh)) - base_h
        out["val_B"] = (float(d.mean()), float(d.std() / np.sqrt(len(d))))
        out["hold"] = (float(dh.mean()), float(dh.std() / np.sqrt(len(dh))))
    m = fit(hA, 50)  # forced fit (no do-nothing option), for the magnitude
    d = (V.ndcg(V.zb + m.predict(Xv)) - base_v)[~hA]; dh = Hd.ndcg(Hd.zb + m.predict(Xh)) - base_h
    out["forced50_val_B"] = (float(d.mean()), float(d.std() / np.sqrt(len(d))))
    out["forced50_hold"] = (float(dh.mean()), float(dh.std() / np.sqrt(len(dh))))
    imp = m.feature_importance("gain"); o = np.argsort(-imp)[:8]
    out["forced50_top"] = [(["zb", "zb_rk"] + [forms[int(c.split("_")[-1])] if c.split("_")[-1].isdigit() and int(c.split("_")[-1]) < len(forms) else c for c in use])[i] for i in o]
    res[nm] = out
    print(nm, json.dumps(out), flush=True)
json.dump({"res": res, "top": forms[:TOPK]}, open(OUT / f"openfe_result{TAG}.json", "w"), indent=1)
