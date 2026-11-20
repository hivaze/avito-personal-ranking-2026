"""Own per-candidate history-cosine statistics and own label-free feed covariates (val). No labels read here.
History index 0 = newest event. Everything that enters the matching is computed on the NEWEST 64 events only."""
import json
import time
from rv_lib import *

t0 = time.time()
d = ROOT / "data/competition/val/item_embeddings"
meta = json.loads((d / "data.meta").read_text())
T = np.memmap(d / "data.memmap", dtype=np.int8, mode="r").reshape(meta["shape"])
ce = np.load(FLAT / "val/cand_emb_idx.npy")
he = np.load(FLAT / "val/hist_emb_idx.npy")
ccat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
hcat = np.load(FLAT / "val/hist_category_id.npy").astype(np.int64)
het = np.load(FLAT / "val/hist_event_type.npy").astype(np.int64)
hes = np.load(FLAT / "val/hist_event_source.npy").astype(np.int64)
hit = np.load(FLAT / "val/hist_item_id.npy")
NEG = -2.0
smax_all = np.full(N, NEG, np.float32)
smax_64 = np.full(N, NEG, np.float32)
smax_16 = np.full(N, NEG, np.float32)
smax_old = np.full(N, NEG, np.float32)      # max cosine over events at index >= 128 (the older half of a capped history)
smax_n128 = np.full(N, NEG, np.float32)     # max cosine over the newest 128
top16 = np.zeros(N, np.float32)             # mean of the 16 largest cosines (fewer if the history is shorter)
top32 = np.zeros(N, np.float32)
n08 = np.zeros(N, np.int16)                 # number of events with cosine >= 0.8
amax = np.full(N, -1, np.int16)             # index of the most similar event
cov = {k: np.zeros(NF, np.float32) for k in ("n_ccat", "top_ccat", "emb_miss", "ccat_in_h64", "h64_ncat", "h64_topcat", "h64_noemb", "h64_distinct",
                                             "h64_src_top", "hall_distinct")}
et64 = np.zeros((NF, 11), np.float32)


def unit(idx):
    x = np.asarray(T[np.maximum(idx, 0)], np.float32)
    x[idx < 0] = 0
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(nrm, 1e-6)


for f in range(NF):
    o, m, ho, h = OFF[f], M[f], HOFF[f], min(H[f], 256)
    ci = ce[o:o + m]
    cc = ccat[o:o + m]
    v, c = np.unique(cc, return_counts=True)
    cov["n_ccat"][f] = len(v)
    cov["top_ccat"][f] = c.max() / m
    cov["emb_miss"][f] = (ci < 0).mean()
    if h == 0:
        continue
    hi = he[ho:ho + h]
    S = unit(ci) @ unit(hi).T
    S[:, hi < 0] = NEG
    S[ci < 0, :] = NEG
    smax_all[o:o + m] = S.max(1)
    amax[o:o + m] = S.argmax(1)
    smax_64[o:o + m] = S[:, :64].max(1)
    smax_16[o:o + m] = S[:, :16].max(1)
    smax_n128[o:o + m] = S[:, :128].max(1)
    if h > 128:
        smax_old[o:o + m] = S[:, 128:].max(1)
    Ss = -np.sort(-np.where(S == NEG, -1.0, S), axis=1)    # events without an embedding count as cosine -1 in the means
    top16[o:o + m] = Ss[:, :16].mean(1)
    top32[o:o + m] = Ss[:, :32].mean(1)
    n08[o:o + m] = (S >= 0.8).sum(1)
    k = min(h, 64)
    hc = hcat[ho:ho + k]
    cov["ccat_in_h64"][f] = np.isin(cc, hc).mean()
    v, c = np.unique(hc, return_counts=True)
    cov["h64_ncat"][f] = len(v)
    cov["h64_topcat"][f] = c.max() / k
    cov["h64_noemb"][f] = (hi[:k] < 0).mean()
    cov["h64_distinct"][f] = len(np.unique(hit[ho:ho + k])) / k
    cov["hall_distinct"][f] = len(np.unique(hit[ho:ho + h])) / h
    cov["h64_src_top"][f] = np.bincount(hes[ho:ho + k]).max() / k
    et64[f] = np.bincount(het[ho:ho + k], minlength=11)[:11] / k
np.savez(OUT / "cos.npz", smax_all=smax_all, smax_64=smax_64, smax_16=smax_16, smax_old=smax_old, smax_n128=smax_n128, top16=top16, top32=top32, n08=n08, amax=amax)
cold = feat("it_hist_all_luo") == 0
ts = fv["timestamp"].to_numpy().astype(np.int64)
tr = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["user_id"])["user_id"].to_numpy()
u, c = np.unique(tr, return_counts=True)
ntr = np.zeros(NF)
ix = np.searchsorted(u, UID)
ok = (ix < len(u)) & (u[np.minimum(ix, len(u) - 1)] == UID) & (UID != 0)
ntr[ok] = c[ix[ok]]
df = pl.DataFrame(dict(M=M.astype(np.float32), logM=np.log(M).astype(np.float32), day=((ts + 10800) // 86400).astype(np.float32), hour=(((ts + 10800) % 86400) // 3600).astype(np.float32),
                       uid0=(UID == 0).astype(np.float32), cold_share=(np.bincount(FID, weights=cold, minlength=NF) / M).astype(np.float32),
                       **cov, **{f"h64_et{t}": et64[:, t] for t in range(1, 11)}, seen=ok.astype(np.float32), log_ntr=np.log1p(ntr).astype(np.float32)))
df.write_parquet(OUT / "cov.parquet")
print("done", round(time.time() - t0), "s", df.shape, "| seen share", ok.mean(), "| cold cand share", cold.mean())
