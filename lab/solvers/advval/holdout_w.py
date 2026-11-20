"""Does an adversarially reweighted train holdout track val better? (61 models with holdout + train-only val preds)
Holdout = nn/train.py --holdout selection (30K of p_random > 0.9 & M >= 50, rng 12345). Weights = density ratio
p(val50)/p(train_rl) from the feed-level 'intrinsic' classifier of comparison d (OOF), clipped to [0.1, 10]."""
import glob, os, json
from common_av import *
from scipy.stats import spearmanr, pearsonr

ftr, Mtr, offtr = meta("train")
pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
pr[Mtr < 50] = 0.0
cand = np.where((pr > 0.9) & (Mtr >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh, offh = np.load(f"{PREDS}/holdout/holdout_meta.npy")
assert (Mh == Mtr[hold]).all()
yh = np.load(f"{PREDS}/holdout/holdout_y.npy")
fv, Mv, offv = meta("val"); yv = np.load(f"{FLAT}/val/cand_target.npy")
v50 = Mv >= 50


def wts(sname, clip=(0.1, 10.0)):
    d = np.load(f"{OUT}/feed_oof_d_trainrl_vs_val50_{sname}.npz")
    p, yy, fidx = d["oof"], d["y"], d["fidx"]
    pa = dict(zip(fidx[yy == 0], p[yy == 0]))
    ph = np.array([pa[i] for i in hold])
    nA, nB = (yy == 0).sum(), (yy == 1).sum()
    w = ph / (1 - ph) * nA / nB; w /= w.mean()
    if clip:
        w = np.clip(w, *clip); w /= w.mean()
    return w


W = {"unweighted": np.ones(len(hold)), "intrinsic": wts("intrinsic")}
ess = {k: float(w.sum() ** 2 / (w ** 2).sum()) for k, w in W.items()}
print("ESS", ess)
rows = []
for f in sorted(glob.glob(f"{PREDS}/holdout/nn_*_holdout.npy")):
    b = os.path.basename(f)[:-len("_holdout.npy")]
    vf = f"{PREDS}/{b}_trainonly_val.npy"
    if not os.path.exists(vf):
        continue
    ph = ndcg_per_feed(np.load(f), yh, offh, Mh)
    pv = ndcg_per_feed(np.load(vf), yv, offv, Mv)
    rows.append(dict(name=b, hold=ph, val=pv))
names = [r["name"] for r in rows]
V = np.array([r["val"].mean() for r in rows]); V50 = np.array([r["val"][v50].mean() for r in rows])
res = {"ess": ess, "n_models": len(rows)}
print("models", len(rows))
for wn, w in W.items():
    Hs = np.array([(r["hold"] * w).sum() / w.sum() for r in rows])
    res[wn] = dict(pearson_val=pearsonr(Hs, V)[0], spearman_val=spearmanr(Hs, V)[0],
                   pearson_val50=pearsonr(Hs, V50)[0], spearman_val50=spearmanr(Hs, V50)[0])
    # pairwise: among pairs with |dval| > 2 SE (paired, all val), share with the same sign of dholdout
    agree, tot, agree_sig, tot_sig = 0, 0, 0, 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dv = rows[i]["val"] - rows[j]["val"]
            se = dv.std(ddof=1) / np.sqrt(len(dv))
            dh = ((rows[i]["hold"] - rows[j]["hold"]) * w).sum() / w.sum()
            if abs(dv.mean()) > 2 * se:
                tot += 1; agree += np.sign(dh) == np.sign(dv.mean())
    res[wn]["pair_sign_agree_sig"] = agree / max(tot, 1); res[wn]["n_sig_pairs"] = tot
    print(wn, {k: round(float(v), 4) for k, v in res[wn].items()})
json.dump(res, open(f"{OUT}/holdout_w.json", "w"), indent=1, default=float)
