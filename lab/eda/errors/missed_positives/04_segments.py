"""Segment analysis: category / cluster novelty of positives, model calibration by novelty class, and predictability of
"exploration" feeds (feeds whose positive is in a category absent from the recent history). Leak-free val feeds only.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

D = Path("<repo>/eda/errors/missed_positives")


def softmax_scale(z, y, fid, nf):
    """Fit a in P(i is a positive) = exp(a z_i) / sum_feed exp(a z_j) (conditional logit, each positive one event)."""
    a = 1.0
    for _ in range(30):
        e = np.exp(a * z - np.repeat(np.maximum.reduceat(a * z, OFF), MM))
        S0 = np.bincount(fid, weights=e, minlength=nf)
        S1 = np.bincount(fid, weights=e * z, minlength=nf)
        S2 = np.bincount(fid, weights=e * z * z, minlength=nf)
        npos = np.bincount(fid, weights=y, minlength=nf)
        m1 = S1 / S0; m2 = S2 / S0
        g = (y * z).sum() - (npos * m1).sum()
        h = -(npos * (m2 - m1 ** 2)).sum()
        st = g / h; a -= st
        if abs(st) < 1e-8:
            break
    e = np.exp(a * z - np.repeat(np.maximum.reduceat(a * z, OFF), MM))
    p = e / np.repeat(np.bincount(fid, weights=e, minlength=nf), MM) * np.repeat(np.bincount(fid, weights=y, minlength=nf), MM)
    return a, p


def main():
    global OFF, MM
    fr = np.load(D / "frame.npz")
    y, best, zc, M, off, leakf, fid, r, q = (fr[k] for k in ["y", "best", "z_clean", "M", "off", "leakf", "fid", "r", "q"])
    OFF, MM = off, M
    nf = len(M)
    keepc = ~leakf[fid]
    L = lambda n: np.load(FEATS / "val" / f"{n}.npy").astype(np.float64)
    Ld = lambda n: np.load(D / "derived/val" / f"{n}.npy").astype(np.float64)
    cmin = L("cat_min_hpos"); ccnt = L("cat_cnt_all"); c20 = L("cat_cnt_20")
    clu = L("clu_hmin1024"); ms = L("maxsim32")
    H = np.repeat(L("H")[off] if False else pl.read_parquet(FLAT / "val/feeds.parquet")["H"].to_numpy(), M).astype(np.float64)
    nov = np.where(ccnt == 0, 0, np.where(cmin >= 32, 1, np.where(cmin >= 8, 2, 3)))
    nov_names = ["cat never in hist", "cat only at hpos>=32", "cat at hpos 8-31", "cat in last 8"]
    pos = y > 0
    a_b, p_b = softmax_scale(best, y, fid, nf)
    a_c, p_c = softmax_scale(zc, y, fid, nf)
    print(f"softmax scale: best {a_b:.3f}, clean {a_c:.3f}")
    print("\n== candidate category novelty (leak-free) ==")
    print(f"{'class':24s} {'cand share':>10s} {'pos share':>9s} {'pos rate':>8s} {'MP rate':>7s} {'mean q_pos':>10s} {'act/exp best':>12s} {'act/exp clean':>13s}")
    for k in range(4):
        m = (nov == k) & keepc
        mp = m & pos
        print(f"{nov_names[k]:24s} {m.sum() / keepc.sum():10.3f} {mp.sum() / (pos & keepc).sum():9.3f} {y[m].mean():8.4f} "
              f"{(q[mp] >= 0.5).mean():7.3f} {q[mp].mean():10.3f} {y[m].sum() / p_b[m].sum():12.3f} {y[m].sum() / p_c[m].sum():13.3f}")
    # bootstrap SE over feeds for act/exp of the novel classes
    rng = np.random.default_rng(0)
    feeds = np.where(~leakf)[0]
    A = np.zeros((4, nf)); E = np.zeros((4, nf)); Ec = np.zeros((4, nf))
    for k in range(4):
        m = (nov == k)
        A[k] = np.bincount(fid, weights=y * m, minlength=nf); E[k] = np.bincount(fid, weights=p_b * m, minlength=nf)
        Ec[k] = np.bincount(fid, weights=p_c * m, minlength=nf)
    bs = []
    for _ in range(300):
        s = rng.choice(feeds, len(feeds))
        bs.append([A[k, s].sum() / E[k, s].sum() for k in range(4)])
    print("bootstrap SE of act/exp (best):", np.round(np.std(bs, 0), 3))
    # similarity buckets
    print("\n== maxsim32 quintile (within leak-free) ==")
    qs = np.nanquantile(ms[keepc], [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(np.nan_to_num(ms, nan=-1), qs)
    for k in range(5):
        m = (b == k) & keepc; mp = m & pos
        print(f"maxsim q{k}: pos share {mp.sum() / (pos & keepc).sum():.3f} pos rate {y[m].mean():.4f} MP rate {(q[mp] >= 0.5).mean():.3f} act/exp best {y[m].sum() / p_b[m].sum():.3f} clean {y[m].sum() / p_c[m].sum():.3f}")
    # cross-tab novelty x f3 rank (f3 = user x category affinity)
    f3rk = L("f3_rk"); f1rk = L("f1_rk")
    print("\n== novel-category positives: f3 within-feed rank (1 = highest) ==")
    for k in range(4):
        mp = (nov == k) & keepc & pos; mn = (nov == k) & keepc & ~pos
        print(f"{nov_names[k]:24s} f3_rk pos {np.nanmean(f3rk[mp]):.3f} neg {np.nanmean(f3rk[mn]):.3f} | f1_rk pos {np.nanmean(f1rk[mp]):.3f} neg {np.nanmean(f1rk[mn]):.3f}")
    # feed-level: exploration feeds
    any_recent = np.bincount(fid, weights=(pos & (nov == 3)).astype(float), minlength=nf) > 0
    all_novel = np.bincount(fid, weights=(pos & (nov <= 1)).astype(float), minlength=nf) == np.bincount(fid, weights=pos.astype(float), minlength=nf)
    per_b = ndcg_per_feed(best, y, off, M)
    kf = ~leakf
    print(f"\nfeeds whose positives are all in categories absent from the last 32 events: {all_novel[kf].mean():.3f}; "
          f"their NDCG {per_b[kf & all_novel].mean():.4f} vs {per_b[kf & ~all_novel].mean():.4f}; share of (1-NDCG) mass "
          f"{(1 - per_b[kf & all_novel]).sum() / (1 - per_b[kf]).sum():.3f}")
    # share of candidates in such categories per feed
    novshare = np.bincount(fid, weights=(nov <= 1).astype(float), minlength=nf) / M
    print(f"mean share of candidates in such categories: {novshare[kf].mean():.3f}; positives novel share {(pos & (nov <= 1) & keepc).sum() / (pos & keepc).sum():.3f}")
    # predictability of exploration feeds from feed-level features (only feed-level info, no labels)
    fnames = ["H", "hist_n_uniq_cat", "ctx_fs_in_hist_share", "ctx_fs_n_cat", "ctx_fs_cat_ent", "M"] + [f"u_het{e}_share" for e in range(1, 11)]
    X = np.stack([L(n)[off] for n in fnames] + [Ld(n)[off] for n in ["d_uemb_norm", "d_h_ncat_ratio", "d_h_et1_32"]] + [novshare], 1)
    users = pl.read_parquet(FLAT / "val/feeds.parquet")["user_id"].to_numpy().copy()
    users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
    Xk, yk, gk = X[kf], all_novel[kf].astype(int), users[kf]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4)
    pr = cross_val_predict(clf, np.nan_to_num(Xk, nan=-1), yk, groups=gk, cv=GroupKFold(5), method="predict_proba")[:, 1]
    print(f"exploration-feed AUC from feed-level features (HGB, 5-fold by user): {roc_auc_score(yk, pr):.3f}; "
          f"base rate {yk.mean():.3f}")
    # only novshare (trivial prior: more novel candidates -> more likely novel positive)
    print(f"  AUC of novel-candidate share alone: {roc_auc_score(yk, Xk[:, -1]):.3f}")
    # normalized: P(pos novel) vs share of novel cands (is novelty over/under-represented among positives?)
    ex_rate = (pos & (nov <= 1) & keepc).sum() / (pos & keepc).sum()
    print(f"positives in novel-ish categories {ex_rate:.3f} vs candidates {((nov <= 1) & keepc).mean():.3f} -> lift {ex_rate / ((nov <= 1) & keepc).mean():.3f}")
    np.save(D / "explore_pred.npy", np.c_[np.where(kf)[0], pr, yk])


if __name__ == "__main__":
    main()
