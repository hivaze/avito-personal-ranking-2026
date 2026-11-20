"""Label-free: how many DISTINCT items fill the 16 retrieval slots (top-16 history events by cosine), by H bucket."""
from lib import *
tix = np.load(OUT / "cos_top64_idx.npy"); hitem = np.load(FLAT / "val/hist_item_id.npy"); hoff = fv["hist_off"].to_numpy().astype(np.int64)
N = len(yv); nd16 = np.zeros(N, np.float32); nd32 = np.zeros(N, np.float32); nv16 = np.zeros(N, np.float32); dupall = np.zeros(NF)
for f in range(NF):
    h = Hv[f]
    if h == 0: continue
    o, m = ov[f], Mv[f]; ids = hitem[hoff[f]:hoff[f] + h]; dupall[f] = 1 - len(np.unique(ids)) / h
    for k, out in ((16, nd16), (32, nd32)):
        ix = tix[o:o + m, :k].astype(np.int64); ok = ix >= 0; g = np.where(ok, ids[np.maximum(ix, 0)], -1 - np.arange(k)[None, :])
        g = np.sort(g, 1); out[o:o + m] = 1 + (g[:, 1:] != g[:, :-1]).sum(1) - (~ok).sum(1) * 1.0 + ((~ok).sum(1) > 0) * 0.0
        if k == 16: nv16[o:o + m] = ok.sum(1)
    # correct: distinct among valid slots only
    for k, out in ((16, nd16), (32, nd32)):
        ix = tix[o:o + m, :k].astype(np.int64); ok = ix >= 0
        g = np.where(ok, ids[np.maximum(ix, 0)], -1); g = np.sort(g, 1); d = 1 + (g[:, 1:] != g[:, :-1]).sum(1); d = d - (~ok).any(1)   # drop the -1 group
        out[o:o + m] = d
Hc = np.repeat(Hv, Mv); y = yv.astype(bool); rows = []
for n, a, b in BUCK[3:]:
    q = (Hc >= a) & (Hc <= b)
    rows.append([n, f"{nv16[q].mean():.1f}", f"{nd16[q].mean():.2f}", f"{(nd16[q] / np.maximum(nv16[q], 1)).mean():.3f}", f"{(nd16[q & y] / np.maximum(nv16[q & y], 1)).mean():.3f}", f"{(nd16[q] <= 8).mean():.3f}", f"{nd32[q].mean():.2f}", f"{dupall[bmask[n]].mean():.3f}"])
print(tab(rows, ["H", "valid slots of 16", "distinct items in the 16 slots", "distinct share (candidates)", "distinct share (positives)", "share of candidates with <= 8 distinct", "distinct items in top-32", "repeat share of the whole stored history"]))
