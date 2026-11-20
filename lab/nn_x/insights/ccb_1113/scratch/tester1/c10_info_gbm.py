"""C10 INFORMATION: scratchpad/ext/xcheck193/inter_resid.py GBM T - C (control C = GBM lambdarank init beta*b on [b + 10 cues];
T = C inputs + user aggregates), nested user-grouped 5-fold on val (folds = harness GroupKFold by user_id) and full-val fit
applied to the 30K holdout (b_h, holdout aggregates as built in c10_eval.py: tte_u_* as-of days < d)."""
import lightgbm as lgb
from pathlib import Path
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(Path(__file__).resolve().parent / "c10_eval.py").read().split("# standardise with val feed-level mean/sd")[0])
P = dict(objective="lambdarank", learning_rate=0.05, num_leaves=7, min_data_in_leaf=2000, lambda_l2=50.0, verbose=-1,
         num_threads=12, lambdarank_truncation_level=300, label_gain=[0, 1], seed=0, deterministic=True)
Qv_raw = np.stack([ldv(c) for c in CUES], 1); Qh_raw = np.stack([np.asarray(ldt(c)[H.hrows]).astype(np.float64) for c in CUES], 1)
def run(extra_v, extra_h):
    Xv = np.c_[H.b, Qv_raw] if extra_v is None else np.c_[H.b, Qv_raw, extra_v]
    Xh = np.c_[H.bh, Qh_raw] if extra_h is None else np.c_[H.bh, Qh_raw, extra_h]
    s = np.zeros(len(H.y))
    for k, (trf, vaf) in enumerate(H.folds):
        tr = H.rowfold != k
        beta = H.fit_lin(H.b[:, None], tr)[0]
        ds = lgb.Dataset(Xv[tr], H.y[tr], group=H.M[np.sort(trf)], init_score=beta * H.b[tr])
        m = lgb.train(P, ds, 60); s[~tr] = beta * H.b[~tr] + m.predict(Xv[~tr], raw_score=True)
    beta = H.fit_lin(H.b[:, None], np.ones(len(H.y), bool))[0]
    m = lgb.train(P, lgb.Dataset(Xv, H.y, group=H.M, init_score=beta * H.b), 60)
    sh = beta * H.bh + m.predict(Xh, raw_score=True)
    return H.nd(s), H.ndh(sh)
cV, cH = run(None, None)
F26 = ["log1pH", "u0", "u_het1_share", "u_het2_share", "u_het3_share", "u_het4_share", "u_het10_share", "tte_u_ctr", "tte_u_imp", "u_contact32", "u_fav32"]
F27 = ["hist_n_uniq_cat", "ent32", "ent32_top", "entall", "entall_top"]
for g, names in (("F26", F26), ("F27", F27), ("F26+F27", F26 + F27)):
    ev = np.stack([np.repeat(A_v[n], H.M) for n in names], 1); eh = np.stack([np.repeat(A_h[n], H.Mh) for n in names], 1)
    tV, tH = run(ev, eh)
    d, dh = tV - cV, tH - cH
    m, se, z = H.stats(d); mh, seh, zh = H.stats(dh)
    log(f"INFO C10 GBM T - C {g:8s}: val {m:+.5f} ({se:.5f}) z {z:+.2f} | holdout {mh:+.5f} ({seh:.5f}) z {zh:+.2f} late {dh[H.H_LATE].mean():+.5f} early {dh[H.H_EARLY].mean():+.5f}")
