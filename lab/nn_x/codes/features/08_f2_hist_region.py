"""Does f2 track the user's activity share in the candidate's region?
Region = K2 component of frequent items (04_k2_geo, out/k2_item_comp.npy). For a fresh train sample (100K uniform feeds, seed 5)
and for val: share of the user's (region-known) history events whose item lies in the candidate's region, vs f2."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *
from scipy.stats import spearmanr

ic = np.load(OUT / "k2_item_comp.npy")
c_items, c_comp = ic[:, 0], ic[:, 1]
o = np.argsort(c_items); c_items = c_items[o]; c_comp = c_comp[o]

def comp_of(items):
    i = np.searchsorted(c_items, items); i = np.minimum(i, len(c_items) - 1)
    return np.where(c_items[i] == items, c_comp[i], -1)

for split, fidx in [("train", train_sample("uniform", 100_000, 5)), ("val", None)]:
    d = load(split, fidx)
    f = feeds(split)
    if fidx is None: fidx = np.arange(len(f))
    hoff = f["hist_off"].to_numpy()[fidx]; H = f["H"].to_numpy()[fidx]
    hrows = np.repeat(hoff - np.cumsum(np.r_[0, H[:-1]]), H) + np.arange(H.sum())
    hitem = np.asarray(np.load(FLAT / split / "hist_item_id.npy", mmap_mode="r")[hrows]).astype(np.int64)
    hfeed = np.repeat(np.arange(len(fidx)), H)
    hc = comp_of(hitem)
    known = hc >= 0
    # counts per (feed, comp)
    kf = hfeed[known].astype(np.int64) * 100000 + hc[known]
    uk, cnt = np.unique(kf, return_counts=True)
    nknown = np.bincount(hfeed[known], minlength=len(fidx))
    cc = comp_of(d["item"].astype(np.int64))
    m = (cc >= 0) & (nknown[d["feed"]] >= 5)
    q = d["feed"][m].astype(np.int64) * 100000 + cc[m]
    j = np.searchsorted(uk, q); j = np.minimum(j, len(uk) - 1)
    c_in = np.where(uk[j] == q, cnt[j], 0)
    share = c_in / nknown[d["feed"][m]]
    f2 = d["f2"][m]
    print(f"\n{split}: candidate rows with region known and >= 5 region-known history events: {m.sum()} ({m.mean():.3f})")
    print(f"   Spearman(f2, history share in candidate's region) = {spearmanr(f2, share).correlation:+.3f}; Spearman(f2, count) = {spearmanr(f2, c_in).correlation:+.3f}")
    print(f"   P(f2 == 0 | share == 0) = {np.mean(f2[share==0]==0):.3f};  P(f2 == 0 | share > 0) = {np.mean(f2[share>0]==0):.3f}")
    for lo, hi in [(0, 1e-9), (1e-9, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        mm = (share >= lo) & (share < hi)
        if mm.sum() > 100:
            print(f"   share in [{lo:.2f},{hi:.2f}): rows {mm.sum()}, f2 mean {f2[mm].mean():.3f}, f2 median {np.median(f2[mm]):.3f}, f2==0 {np.mean(f2[mm]==0):.3f}")
    # within feed: rank agreement between f2 and share across the candidate's regions
    # feeds with >= 2 distinct candidate regions
    df = pl.DataFrame({"feed": d["feed"][m], "comp": cc[m], "f2": f2, "share": share})
    g = df.group_by(["feed", "comp"]).agg(pl.col("f2").first(), pl.col("f2").n_unique().alias("nu"), pl.col("share").first())
    print(f"   (feed, region) groups: {len(g)}; share with a single f2 value inside the group {np.mean(g['nu'].to_numpy()==1):.4f}")
    g2 = g.filter(pl.col("feed").count().over("feed") >= 2)
    # concordance of pairs within feed
    a = g2.sort(["feed", "comp"])
    fe = a["feed"].to_numpy(); v = a["f2"].to_numpy(); s = a["share"].to_numpy()
    conc = disc = 0
    st = np.flatnonzero(np.r_[True, fe[1:] != fe[:-1]]); en = np.r_[st[1:], len(fe)]
    for s0, e0 in zip(st, en):
        vv = v[s0:e0]; ss = s[s0:e0]
        dv = np.sign(vv[:, None] - vv[None, :]); ds = np.sign(ss[:, None] - ss[None, :])
        prod = dv * ds
        conc += np.sum(prod > 0); disc += np.sum(prod < 0)
    print(f"   within-feed region pairs: concordant {conc//2}, discordant {disc//2}, Kendall-like {(conc-disc)/(conc+disc):+.3f}")
