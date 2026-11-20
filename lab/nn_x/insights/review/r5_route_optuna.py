"""Review R5: nested (user-grouped 5-fold, outer) evaluation of FT-weight routing rules for G3 / PF7x / M2, with an
Optuna TPE search (60 trials) over (M threshold T, w_ft for M<T, w_ft for M>=T) inside each outer training fold.
Compares out-of-fold: fixed 0.5 (current), the pre-declared M-split (T=50, 0 / 0.5), and the Optuna-tuned rule.
Per-feed NDCG is precomputed on a w_ft grid (0..1 step 0.05) so every rule is a per-feed lookup."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"): os.environ[k] = "12"
import sys; sys.path.insert(0, "<repo>/src")
from common import *
import optuna
from sklearn.model_selection import GroupKFold
optuna.logging.set_verbosity(optuna.logging.WARNING)
P = Path("<repo>/preds")
fv, Mv, offv = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy"); fid = np.repeat(np.arange(len(Mv)), Mv)
uid = fv["user_id"].to_numpy(); grp = np.where(uid == 0, -1 - np.arange(len(uid)), uid)
def z(s):
    s = s.astype(np.float64); n = len(Mv)
    mu = np.bincount(fid, s, n) / Mv; sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / Mv - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]
W = np.round(np.arange(0, 1.0001, 0.05), 2)
for tag in ("G3", "PF7x", "M2"):
    zf, zt = z(np.load(P / f"ens_{tag}_ft_val.npy")), z(np.load(P / f"ens_{tag}_to_val.npy"))
    N = np.stack([ndcg_per_feed((w * zf + (1 - w) * zt).astype(np.float32), yv, offv, Mv) for w in W], 1)  # feeds x grid
    wi = lambda w: int(round(w / 0.05))
    def rule(T, ws, wl, idx): return np.where(Mv[idx] < T, N[idx, wi(ws)], N[idx, wi(wl)])
    oof_base, oof_ms, oof_opt, chosen = [], [], [], []
    for k, (tr, te) in enumerate(GroupKFold(5).split(np.arange(len(Mv)), groups=grp)):
        def obj(t):
            T = t.suggest_int("T", 10, 150, step=5); ws = t.suggest_float("ws", 0, 1, step=0.05); wl = t.suggest_float("wl", 0, 1, step=0.05)
            return rule(T, ws, wl, tr).mean()
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=k))
        st.enqueue_trial({"T": 50, "ws": 0.0, "wl": 0.5}); st.enqueue_trial({"T": 50, "ws": 0.5, "wl": 0.5})
        st.optimize(obj, n_trials=60)
        bp = st.best_params; chosen.append(bp)
        oof_base.append(N[te, wi(0.5)]); oof_ms.append(rule(50, 0.0, 0.5, te)); oof_opt.append(rule(bp["T"], bp["ws"], bp["wl"], te))
        # in-fold optimism of the tuned rule
    b, m, o = map(np.concatenate, (oof_base, oof_ms, oof_opt))
    se = lambda d: d.std() / np.sqrt(len(d))
    print(f"{tag}: OOF fixed0.5 {b.mean():.5f} | M-split {m.mean()-b.mean():+.5f}+-{se(m-b):.5f} | Optuna-tuned rule {o.mean()-b.mean():+.5f}+-{se(o-b):.5f} | tuned-Msplit {o.mean()-m.mean():+.5f}+-{se(o-m):.5f}")
    print("   chosen per fold:", [(c["T"], c["ws"], c["wl"]) for c in chosen])
    # global w (no routing) nested, for reference
    gw = []
    for k, (tr, te) in enumerate(GroupKFold(5).split(np.arange(len(Mv)), groups=grp)):
        j = N[tr].mean(0).argmax(); gw.append(N[te, j])
    g = np.concatenate(gw); print(f"   nested global w_ft: {g.mean()-b.mean():+.5f}+-{se(g-b):.5f}")
