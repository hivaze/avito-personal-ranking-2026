"""Label-free per-feed covariates + per-candidate history-cosine statistics for val (CPU). Writes out/feed_cov.parquet, out/cos_*.npy."""
from lib import *
import json, time
t0 = time.time()
d = ROOT / "data/competition/val/item_embeddings"; meta = json.loads((d / "data.meta").read_text())
T = np.fromfile(d / "data.memmap", dtype=np.int8).reshape(meta["shape"])
ce_i = np.load(FLAT / "val/cand_emb_idx.npy"); he_i = np.load(FLAT / "val/hist_emb_idx.npy")
ccat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64); hcat = np.load(FLAT / "val/hist_category_id.npy").astype(np.int64)
het = np.load(FLAT / "val/hist_event_type.npy").astype(np.int64); hes = np.load(FLAT / "val/hist_event_source.npy").astype(np.int64)
hitem = np.load(FLAT / "val/hist_item_id.npy"); citem = np.load(FLAT / "val/cand_item_id.npy")
ue = np.load(FLAT / "val/user_emb.npy").astype(np.float32)
hoff = fv["hist_off"].to_numpy().astype(np.int64)
N = len(ce_i); KT = 64
top = np.full((N, KT), -2.0, np.float16); tix = np.full((N, KT), -1, np.int16)
WIN = [8, 16, 32, 64, 128, 256]
smaxw = np.full((N, len(WIN)), -1.5, np.float32)       # max cosine within the newest w events
cos_u = np.zeros(N, np.float32); smean = np.zeros(N, np.float32)
nge = np.zeros((N, 3), np.int16)                        # number of events with cosine >= 0.9 / 0.8 / 0.7
initem = np.zeros(N, np.int16)                          # occurrences of the candidate's own item id in the history
initem_first = np.full(N, -1, np.int16)                 # newest index of the candidate's own item in the history
cov = {k: np.zeros(NF, np.float32) for k in ["emb_miss_share", "n_ccat", "top_ccat_share", "ccat_in_hist_share", "h64_ncat", "h64_topcat", "h64_noemb", "h64_distinct",
                                             "h64_src_top", "hall_ncat", "hall_topcat", "hall_distinct", "hall_n_emb", "ue_norm", "h64_et1_run", "cand_in_hist_share"]}
et64 = np.zeros((NF, 11), np.float32); etall = np.zeros((NF, 11), np.float32)
for f in range(NF):
    o, m, ho, h = ov[f], Mv[f], hoff[f], Hv[f]
    ci = ce_i[o:o + m]; cm = ci < 0
    C = T[np.maximum(ci, 0)].astype(np.float32); C[cm] = 0
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-3)
    u = ue[f]; un = np.linalg.norm(u); cov["ue_norm"][f] = un
    cos_u[o:o + m] = C @ (u / max(un, 1e-3))
    cc = ccat[o:o + m]; vals, cnts = np.unique(cc, return_counts=True)
    cov["emb_miss_share"][f] = cm.mean(); cov["n_ccat"][f] = len(vals); cov["top_ccat_share"][f] = cnts.max() / m
    if h == 0:
        continue
    hi = he_i[ho:ho + h]; hm = hi < 0
    Hm = T[np.maximum(hi, 0)].astype(np.float32); Hm[hm] = 0
    Hm /= np.maximum(np.linalg.norm(Hm, axis=1, keepdims=True), 1e-3)
    S = C @ Hm.T                                       # [m, h]
    S[:, hm] = -1.5; S[cm, :] = -1.5
    for j, w in enumerate(WIN):
        smaxw[o:o + m, j] = S[:, :w].max(1)
    ok = ~hm; nh = max(int(ok.sum()), 1)
    smean[o:o + m] = np.where(cm, 0, (S * ok[None, :]).sum(1) / nh)
    for j, th in enumerate((0.9, 0.8, 0.7)):
        nge[o:o + m, j] = (S >= th).sum(1)
    k = min(KT, h); idx = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < h else np.tile(np.arange(h), (m, 1))
    v = np.take_along_axis(S, idx, 1); od = np.argsort(-v, axis=1, kind="stable")
    top[o:o + m, :k] = np.take_along_axis(v, od, 1); tix[o:o + m, :k] = np.take_along_axis(idx, od, 1)
    hc = hcat[ho:ho + h]; cov["ccat_in_hist_share"][f] = np.isin(cc, hc).mean()
    hid = hitem[ho:ho + h]; eqi = citem[o:o + m, None] == hid[None, :]
    initem[o:o + m] = eqi.sum(1); a = eqi.argmax(1); initem_first[o:o + m] = np.where(eqi.any(1), a, -1)
    cov["cand_in_hist_share"][f] = eqi.any(1).mean()
    for nm, sl in (("h64", slice(0, min(h, 64))), ("hall", slice(0, h))):
        c_ = hc[sl]; v_, n_ = np.unique(c_, return_counts=True)
        cov[nm + "_ncat"][f] = len(v_); cov[nm + "_topcat"][f] = n_.max() / len(c_)
        cov[nm + "_distinct"][f] = len(np.unique(hid[sl])) / len(c_)
        et = np.bincount(het[ho:ho + h][sl], minlength=11)[:11] / len(c_)
        (et64 if nm == "h64" else etall)[f] = et
    cov["h64_noemb"][f] = hm[:64].mean(); cov["hall_n_emb"][f] = ok.sum()
    s_ = hes[ho:ho + min(h, 64)]; cov["h64_src_top"][f] = np.bincount(s_).max() / len(s_)
    if f % 3000 == 0: print(f, time.time() - t0, flush=True)
np.save(OUT / "cos_top64.npy", top); np.save(OUT / "cos_top64_idx.npy", tix); np.save(OUT / "cos_smaxw.npy", smaxw)
np.save(OUT / "cos_u.npy", cos_u); np.save(OUT / "cos_smean.npy", smean); np.save(OUT / "cos_nge.npy", nge)
np.save(OUT / "cand_initem.npy", initem); np.save(OUT / "cand_initem_first.npy", initem_first)
cold = np.nan_to_num(np.load(FC / "it_hist_all_luo.npy").astype(np.float64), nan=0.0) == 0
ts = fv["timestamp"].to_numpy().astype(np.int64)
df = pl.DataFrame(dict(fidx=np.arange(NF), M=Mv, H=Hv, uid0=(uid == 0), day=(ts + 10800) // 86400, hour=((ts + 10800) % 86400) // 3600,
                       cold_share=np.bincount(fid, weights=cold, minlength=NF) / Mv, **cov,
                       **{f"h64_et{t}": et64[:, t] for t in range(1, 11)}, **{f"hall_et{t}": etall[:, t] for t in range(1, 11)}))
df.write_parquet(OUT / "feed_cov.parquet"); np.save(OUT / "cand_cold.npy", cold)
print("done", time.time() - t0, df.shape)
