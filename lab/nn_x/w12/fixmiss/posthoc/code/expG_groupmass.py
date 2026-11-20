"""(e1) Feed-level GROUP-MASS recalibration (untested form: 'why users click outside their recent categories').
For a binary item group g (new-to-user category / cold item / not in the top item's category), per feed:
  model mass m_f = sum_{i in g} softmax(s)_i ; actual share a_f = sum_{i in g} y_i / K_f.
A LightGBM cross-entropy model on half A predicts a_f from logit(m_f) + label-free feed descriptors; on half B every
item of g gets the same shift delta_f = lam * (logit(a_hat_f) - logit(m_f)) (two-group softmax shift: within-group order
kept, only the interleaving of the two groups changes). Reverse halves. lam 1 declared, 0.5 information.
Also the zero-information check: a_hat from logit(m_f) alone (pure mass recalibration curve)."""
import ph, numpy as np, polars as pl, lightgbm as lgb, json
from feedfeat import feed_features
D = ph.Data("val"); fid = D.fid; s = D.s; n = D.n
F = pl.read_parquet("../out/valfeat.parquet", columns=["it_hist_all_luo", "cat_hist_share", "cat", "newest_type", "newest_src", "n_hist_cats32", "hour", "weekday", "guest", "same_cat_newest", "cat_views", "cat_actions"])
mx = np.maximum.reduceat(s, D.off); e = np.exp(s - mx[fid]); q = e / np.bincount(fid, e, n)[fid]
cat = F["cat"].to_numpy(); top = np.zeros(n, np.int64)
for f in range(n):
    o, m = D.off[f], D.M[f]; top[f] = cat[o + np.argmax(s[o:o + m])]
groups = {"newcat": (np.nan_to_num(F["cat_hist_share"].to_numpy(), nan=0) <= 0),
          "cold": (F["it_hist_all_luo"].to_numpy() <= 0),
          "offtopcat": (cat != top[fid])}
X0 = feed_features("val", D); fx = {k: v for k, v in X0.items()}
for c in ("newest_type", "newest_src", "n_hist_cats32", "hour", "weekday", "guest"):
    fx[c] = F[c].to_numpy()[D.off].astype(float)
res = {}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
for gname, g in groups.items():
    g = g.astype(float)
    mf = np.bincount(fid, q * g, n); af = np.bincount(fid, D.y * g, n) / D.K; share = np.bincount(fid, g, n) / D.M
    ok = (share > 0) & (share < 1)
    print(f"[{gname}] feeds with both groups {ok.mean():.3f}; mean model mass {mf[ok].mean():.4f} vs actual share {af[ok].mean():.4f}")
    Xf = np.column_stack([lg(mf), share] + [fx[k] for k in fx]); names = ["logit_m", "share"] + list(fx)
    for form in ("mass_only", "full"):
        cols = [0] if form == "mass_only" else list(range(Xf.shape[1]))
        for lam in (1.0, 0.5):
            out = s.copy()
            for h in (0, 1):
                A = (D.half == h) & ok; B = (D.half != h) & ok
                prm = dict(objective="cross_entropy", learning_rate=0.03, num_leaves=7, min_data_in_leaf=300, lambda_l2=10.0,
                           feature_fraction=0.9, verbose=-1, num_threads=12, seed=3)
                b = lgb.train(prm, lgb.Dataset(Xf[A][:, cols], af[A]), num_boost_round=200 if form == "full" else 100)
                ah = b.predict(Xf[B][:, cols])
                delta = np.zeros(n); delta[B] = lam * (lg(ah) - lg(mf[B]))
                rr = (D.halfr != h)
                out[rr] = s[rr] + delta[fid][rr] * g[rr]
                if lam == 1.0:
                    print(f"   {gname} {form} half {h}: delta sd {delta[B].std():.3f}, mean {delta[B].mean():+.3f}; CE model vs identity on B: "
                          f"{-(af[B] * np.log(np.clip(ah,1e-6,1)) + (1 - af[B]) * np.log(np.clip(1 - ah,1e-6,1))).mean():.4f} vs "
                          f"{-(af[B] * np.log(np.clip(mf[B],1e-6,1)) + (1 - af[B]) * np.log(np.clip(1 - mf[B],1e-6,1))).mean():.4f}")
            res[f"{gname}_{form}_l{lam}"] = D.report(out, f"G {gname} {form} lam {lam}")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'line'} for k, v in res.items()}, open("../out/expG.json", "w"), indent=1)
