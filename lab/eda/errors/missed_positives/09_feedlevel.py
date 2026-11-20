"""Feed-level view: which feeds/users carry the missed positives?  Leak-free feeds only.
Per feed: has_MP (>=1 positive in the bottom half), NDCG (v5). Compared across feed-level features: AUC of the feature for
has_MP, plus NDCG / MP-rate by quartile. Also: event-type mix of the 32 newest events for MP vs TP3 feeds, and how the
positive's category ranks by the model's best candidate of that category ("category miss" vs "within-category miss").
"""
import sys
sys.path.insert(0, "<repo>/src")
from common import *
from sklearn.metrics import roc_auc_score

D = Path("<repo>/eda/errors/missed_positives")


def main():
    fr = np.load(D / "frame.npz")
    y, best, M, off, leakf, fid, r, q, MP, TP3 = (fr[k] for k in ["y", "best", "M", "off", "leakf", "fid", "r", "q", "MP", "TP3"])
    nf = len(M); kf = ~leakf
    f = pl.read_parquet(FLAT / "val/feeds.parquet")
    ccat = np.load(FLAT / "val/cand_category_id.npy")
    per = ndcg_per_feed(best, y, off, M)
    has_mp = np.bincount(fid, weights=MP.astype(float), minlength=nf) > 0
    def L(n):
        pth = FEATS / "val" / f"{n}.npy"
        if not pth.exists():
            pth = D / "derived/val" / f"{n}.npy"
        return np.load(pth).astype(np.float64)[off] if pth.exists() else None
    fcat_n = np.array([np.unique(ccat[o:o + m]).size for o, m in zip(off, M)], float)
    fcat_max = np.array([np.bincount(ccat[o:o + m]).max() / m for o, m in zip(off, M)], float)
    Ld = lambda n: np.load(D / "derived/val" / f"{n}.npy").astype(np.float64)[off]
    feats = {"M": M.astype(float), "H": f["H"].to_numpy().astype(float), "hist_n_uniq_cat": L("hist_n_uniq_cat"),
             "n feed categories": fcat_n, "max feed category share": fcat_max,
             "uemb_norm": Ld("d_uemb_norm"), "et1 share last32": Ld("d_h_et1_32"), "hist ncat ratio": Ld("d_h_ncat_ratio"),
             "u_het1_share": L("u_het1_share"), "u_het2_share": L("u_het2_share"), "u_het3_share": L("u_het3_share"),
             "u_het4_share": L("u_het4_share"), "u_tr_feeds": L("u_tr_feeds"),
             "user_id==0": (f["user_id"].to_numpy() == 0).astype(float), "hour": L("hour"),
             "n positives": np.bincount(fid, weights=y, minlength=nf)}
    print(f"leak-free feeds {kf.sum()}, with MP {has_mp[kf].sum()} ({has_mp[kf].mean():.3f})")
    print(f"{'feature':28s} {'AUC(has_MP)':>11s} | NDCG by quartile (q1..q4) | MP-rate by quartile")
    for n, x in feats.items():
        if x is None:
            print(f"{n:28s} (file deleted)"); continue
        xk = x[kf]; hk = has_mp[kf]; pk = per[kf]
        ok = np.isfinite(xk)
        a = roc_auc_score(hk[ok], xk[ok]) if np.unique(xk[ok]).size > 1 else np.nan
        qs = np.unique(np.nanquantile(xk, [0.25, 0.5, 0.75]))
        b = np.digitize(xk, qs)
        nd = " ".join(f"{pk[b == k].mean():.3f}" for k in range(len(qs) + 1))
        mr = " ".join(f"{hk[b == k].mean():.3f}" for k in range(len(qs) + 1))
        print(f"{n:28s} {a:11.3f} | {nd:30s} | {mr}")
    # M-normalized: MP is mechanically likelier in big feeds (more room); NDCG by M bin
    print("\nNDCG / MP rate by M bin:")
    for lo, hi in [(5, 20), (20, 50), (50, 100), (100, 200), (200, 301)]:
        m = kf & (M >= lo) & (M < hi)
        print(f"  M {lo:3d}-{hi:3d}: feeds {m.sum():5d} NDCG {per[m].mean():.4f} MP-rate {has_mp[m].mean():.3f}")
    # category-level vs within-category miss for each MP
    n_cat_miss = 0; n_within = 0; rows = []
    for i in np.where(MP)[0]:
        fi = fid[i]; o, m = off[fi], M[fi]
        cats = ccat[o:o + m]; rr = r[o:o + m]
        c = ccat[i]
        best_rank_c = rr[cats == c].min()           # model rank of the best candidate of the positive's category
        ncat_ahead = len(np.unique(cats[rr < best_rank_c]))  # categories whose best candidate is above
        rank_in_cat = (rr[cats == c] < r[i]).sum()  # same-category candidates above the positive
        rows.append((best_rank_c, ncat_ahead, rank_in_cat, (cats == c).sum(), m))
    R = np.array(rows, float)
    print(f"\nMP positives: {len(R)}")
    print(f"  model rank of best candidate of MP's category: median {np.median(R[:, 0]):.0f}, share in top-3 {np.mean(R[:, 0] < 3):.3f}, top-10 {np.mean(R[:, 0] < 10):.3f}")
    print(f"  same-category candidates ranked above the MP: median {np.median(R[:, 2]):.0f}; share of MPs with >=1 such {np.mean(R[:, 2] >= 1):.3f}")
    print(f"  MP's category size in feed: median {np.median(R[:, 3]):.0f}; singleton category share {np.mean(R[:, 3] == 1):.3f}")
    # 'category miss' = category's best candidate itself in the bottom half; 'within-category miss' = category represented in top 10
    cm = R[:, 0] >= R[:, 4] / 2
    print(f"  category entirely in bottom half (category-level miss): {cm.mean():.3f}; category has a top-10 candidate (within-category miss): {np.mean(R[:, 0] < 10):.3f}")


if __name__ == "__main__":
    main()
