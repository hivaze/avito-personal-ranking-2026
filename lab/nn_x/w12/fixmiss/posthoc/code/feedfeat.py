"""Label-free per-feed descriptors for gating (val and test)."""
import ph, numpy as np, polars as pl
def feed_features(split, D=None):
    D = D or ph.Data(split); fid = D.fid; n = D.n
    C = np.load(f"../out/comp_{split}.npz")
    F = pl.read_parquet(f"../out/{split}feat.parquet", columns=["it_hist_all_luo", "xp_imp_c50", "cat_hist_share", "uvec_missing", "guest", "n_feed_cats", "H"])
    def fcorr(a, b):
        a = ph.zf(a, D.off, D.M); b = ph.zf(b, D.off, D.M); return np.bincount(fid, a * b, n) / D.M
    s = D.s
    mx = np.maximum.reduceat(s, D.off); e = np.exp(s - mx[fid]); q = e / np.bincount(fid, e, n)[fid]
    ent = -np.bincount(fid, q * np.log(q + 1e-12), n) / np.log(np.maximum(D.M, 2))
    srt = np.sort(s)  # not per feed; compute top gap per feed
    top1 = np.maximum.reduceat(s, D.off)
    s2 = np.where(s == top1[fid], -np.inf, s); top2 = np.maximum.reduceat(s2, D.off)
    cold = np.bincount(fid, (F["it_hist_all_luo"].to_numpy() <= 0).astype(float), n) / D.M
    newcat = np.bincount(fid, (np.nan_to_num(F["cat_hist_share"].to_numpy(), nan=0) <= 0).astype(float), n) / D.M
    X = dict(logM=np.log(D.M), logH=np.log1p(F["H"].to_numpy()[D.off]), dis_br=1 - fcorr(D.ft, D.to),
             dis_pl=1 - 0.5 * (fcorr(C["ft_base"], C["ft_pl"]) + fcorr(C["to_base"], C["to_pl"])),
             ent=ent, gap12=top1 - top2, cold=cold, newcat=newcat, uvm=F["uvec_missing"].to_numpy()[D.off].astype(float),
             ncat=np.log(F["n_feed_cats"].to_numpy()[D.off]))
    return X
