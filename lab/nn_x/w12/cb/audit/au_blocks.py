"""audit 3b: NEW input blocks a tree does not have today (none of them uses a label or an NN output), for the sets V (val), H (holdout), T (test).
 emb  : raw candidate item embedding, 128 int8 dims
 seq  : per-history-event features (58): cos to each of the 12 newest events, event type + source of those events, the 5 most similar
        events (cos, history position, event type, source), counts of events with cos > 0.8 / 0.9
 prod : element-wise products of the normalised candidate embedding with the normalised user embedding and with the normalised mean
        history embedding (2 x 128, float16): lets a tree re-weight the dot product per dimension (a learned metric)
Low thread count on purpose (light process)."""
import time
from au_common import *
sys.path.insert(0, str(ROOT / "nn_x/cb26"))
import polars as pl
from cbdata import hold_index
FLAT = ROOT / "data/flat"; COMP = ROOT / "data/competition"
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
K = 12


def nrm(E):
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)


def build(tag):
    if (CACHE / f"blk_{tag}_seq.npy").exists(): return
    flat = dict(V="val", H="train", T="test")[tag]
    mt = json.loads((COMP / flat / "item_embeddings/data.meta").read_text())
    table = np.memmap(COMP / flat / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(mt["shape"]))
    f = pl.read_parquet(FLAT / flat / "feeds.parquet")
    if tag == "H": hold, _ = hold_index(); f = f[hold]
    M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
    co = f["cand_off"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy().astype(np.int64)
    ce = np.load(FLAT / flat / "cand_emb_idx.npy", mmap_mode="r"); he = np.load(FLAT / flat / "hist_emb_idx.npy", mmap_mode="r")
    het = np.load(FLAT / flat / "hist_event_type.npy", mmap_mode="r"); hsr = np.load(FLAT / flat / "hist_event_source.npy", mmap_mode="r")
    ue = np.load(FLAT / flat / "user_emb.npy", mmap_mode="r")
    if tag == "H": fidx = f["fidx"].to_numpy().astype(np.int64)
    else: fidx = np.arange(len(M))
    n = int(M.sum()); EMB = np.zeros((n, 128), np.int8); SEQ = np.full((n, 58), np.nan, np.float32); PROD = np.zeros((n, 256), np.float16)
    o = 0
    for k in range(len(M)):
        m_, h_ = M[k], H[k]
        ci = np.asarray(ce[co[k]:co[k] + m_]); Ec = np.asarray(table[np.maximum(ci, 0)]); Ec[ci < 0] = 0
        EMB[o:o + m_] = Ec; Ecn = nrm(Ec.astype(np.float32))
        u = np.asarray(ue[fidx[k]], np.float32); un = u / max(np.linalg.norm(u), 1e-6)
        PROD[o:o + m_, :128] = Ecn * un
        if h_ > 0:
            hi = np.asarray(he[ho[k]:ho[k] + h_]); Eh = np.asarray(table[np.maximum(hi, 0)]).astype(np.float32); Eh[hi < 0] = 0; Ehn = nrm(Eh)
            hm = Ehn.mean(0); PROD[o:o + m_, 128:] = Ecn * (hm / max(np.linalg.norm(hm), 1e-6))
            S = Ecn @ Ehn.T; et = np.asarray(het[ho[k]:ho[k] + h_]); sr = np.asarray(hsr[ho[k]:ho[k] + h_])
            kk = min(K, h_); SEQ[o:o + m_, :kk] = S[:, :kk]
            SEQ[o:o + m_, K:K + kk] = et[:kk]; SEQ[o:o + m_, 2 * K:2 * K + kk] = sr[:kk]
            t5 = min(5, h_); idx = np.argsort(-S, axis=1)[:, :t5]
            SEQ[o:o + m_, 36:36 + t5] = np.take_along_axis(S, idx, 1); SEQ[o:o + m_, 41:41 + t5] = idx
            SEQ[o:o + m_, 46:46 + t5] = et[idx]; SEQ[o:o + m_, 51:51 + t5] = sr[idx]
            SEQ[o:o + m_, 56] = (S > 0.8).sum(1); SEQ[o:o + m_, 57] = (S > 0.9).sum(1)
        o += m_
        if k % 5000 == 0: log(tag, k, len(M))
    np.save(CACHE / f"blk_{tag}_emb.npy", EMB); np.save(CACHE / f"blk_{tag}_seq.npy", SEQ); np.save(CACHE / f"blk_{tag}_prod.npy", PROD)
    log(tag, "saved", EMB.shape, SEQ.shape, PROD.shape)


SEQ_NAMES = [f"sq_cos_n{k}" for k in range(K)] + [f"sq_et_n{k}" for k in range(K)] + [f"sq_src_n{k}" for k in range(K)] + \
            [f"sq_top{k}_cos" for k in range(5)] + [f"sq_top{k}_hpos" for k in range(5)] + [f"sq_top{k}_et" for k in range(5)] + [f"sq_top{k}_src" for k in range(5)] + ["sq_n_cos80", "sq_n_cos90"]
if __name__ == "__main__":
    for tag in sys.argv[1:]: build(tag)
    log("DONE")
