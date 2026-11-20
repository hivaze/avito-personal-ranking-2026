# Builder of the history-competition columns (ctx_hn_*): for each of the user's last 32 history items the
# candidates of the feed compete by cosine similarity, and a candidate collects the recency-weighted share or
# reciprocal rank it wins. No labels and no other feeds are read, so train, val and test rows are built alike.
# Usage as in the docstring: python build_ctx3.py <splits> <variants: base,_d7,_d95,_e1> [threads]
# (no log of the exact call was kept). It imports the loaders from lab/feats_v2/ctx_/build_ctx.py.
# It wrote data/feats/{train,val,test}/ctx_hn_*.npy; six of these columns are inputs of the final ranker.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/feats_v2/ctx_/build_ctx3.py
"""ctx_hn_: competition among candidates for the user's recent history items (reciprocal nearest neighbours).

For the last K=32 non-missing history items h_j (j = recency order) and the feed's candidates c_i:
  ctx_hn_top1      # history items whose most similar candidate in the feed is c_i
  ctx_hn_top1w     same, recency-weighted (0.85^j)
  ctx_hn_top3w     recency-weighted count of history items for which c_i is among the 3 most similar candidates
  ctx_hn_rrw       recency-weighted mean reciprocal rank of c_i among candidates for each history item (1/(1+rank))
  ctx_hn_bestrk    rank of c_i among candidates for its own best-matching history item (0 = c_i is that item's NN)
  ctx_hn_margin    maxsim(c_i) - (best similarity of any OTHER candidate to c_i's best-matching history item)
  ctx_hn_soft05/02 recency-weighted softmax attention mass of each history item over the feed's candidates
                   (temperature 0.05 / 0.02), times M (1 = uniform share)
Usage: python build_ctx3.py <splits> <variants: base,_d7,_d95,_e1> [threads]
"""
import sys, time
import numba as nb
import numpy as np

sys.path.insert(0, "<repo>/feats_v2/ctx_")
from build_ctx import load_meta, load_item_emb, save, FLAT

K = 32
NAMES = ["ctx_hn_top1", "ctx_hn_top1w", "ctx_hn_top3w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin",
         "ctx_hn_soft05", "ctx_hn_soft02"]


@nb.njit(parallel=True, fastmath=True, cache=True)
def _hn_kernel(cand_off, M, hist_off, H, cemb, hemb, hmask, table, K, decay, out):
    D = table.shape[1]
    for f in nb.prange(len(M)):
        co = cand_off[f]; m = M[f]
        Hm = np.zeros((K, D), np.float32)  # K is a runtime argument
        nk = 0
        ho = hist_off[f]
        for j in range(H[f]):
            ei = hemb[ho + j]
            if ei < 0 or not hmask[ho + j]:
                continue
            v = table[ei].astype(np.float32)
            Hm[nk] = v / (np.sqrt((v * v).sum()) + 1e-6)
            nk += 1
            if nk >= K:
                break
        V = np.zeros((m, D), np.float32)
        ok = np.zeros(m, np.bool_)
        for i in range(m):
            ei = cemb[co + i]
            if ei >= 0:
                v = table[ei].astype(np.float32)
                V[i] = v / (np.sqrt((v * v).sum()) + 1e-6)
                ok[i] = True
        if nk == 0:
            for i in range(m):
                for c in range(8):
                    out[co + i, c] = np.nan
            continue
        S = V @ Hm[:nk].T  # m x nk
        for i in range(m):
            if not ok[i]:
                for j in range(nk):
                    S[i, j] = -9.0
        top1 = np.zeros(m); top1w = np.zeros(m); top3w = np.zeros(m); rrw = np.zeros(m)
        s05 = np.zeros(m); s02 = np.zeros(m)
        wsum = 0.0
        # per history item: ranks of candidates
        R = np.zeros((m, nk), np.int32)
        best_other = np.zeros((m, nk), np.float32)  # best similarity among other candidates
        for j in range(nk):
            w = decay ** j
            wsum += w
            col = S[:, j].copy()
            o = np.argsort(-col)
            for r in range(m):
                R[o[r], j] = r
            b1 = col[o[0]]
            b2 = col[o[1]] if m > 1 else -9.0
            for i in range(m):
                best_other[i, j] = b2 if i == o[0] else b1
            i0 = o[0]
            top1[i0] += 1; top1w[i0] += w
            for r in range(min(3, m)):
                top3w[o[r]] += w
            for i in range(m):
                rrw[i] += w / (1.0 + R[i, j])
            z05 = 0.0; z02 = 0.0
            for i in range(m):
                if ok[i]:
                    z05 += np.exp((col[i] - b1) / 0.05); z02 += np.exp((col[i] - b1) / 0.02)
            for i in range(m):
                if ok[i]:
                    s05[i] += w * np.exp((col[i] - b1) / 0.05) / z05
                    s02[i] += w * np.exp((col[i] - b1) / 0.02) / z02
        for i in range(m):
            if not ok[i]:
                for c in range(8):
                    out[co + i, c] = np.nan
                continue
            jb = 0
            for j in range(1, nk):
                if S[i, j] > S[i, jb]:
                    jb = j
            out[co + i, 0] = top1[i]
            out[co + i, 1] = top1w[i] / wsum
            out[co + i, 2] = top3w[i] / wsum
            out[co + i, 3] = rrw[i] / wsum
            out[co + i, 4] = R[i, jb] / (m - 1) if m > 1 else 0.0
            out[co + i, 5] = S[i, jb] - best_other[i, jb]
            out[co + i, 6] = s05[i] / wsum * m
            out[co + i, 7] = s02[i] / wsum * m


# variants: suffix -> (K, decay, event filter or 0 = all, columns to keep)
VARIANTS = {
    "": (32, 0.85, 0, [0, 1, 2, 3, 4, 5]),
    "soft": (32, 0.85, 0, [6, 7]),
    "_d7": (32, 0.7, 0, [1, 3]),
    "_d95": (64, 0.95, 0, [1, 3]),
    "_e1": (32, 0.85, 1, [1, 3, 5]),
}


def build(split, variants):
    t = time.time()
    feeds, M, H, co, ho = load_meta(split)
    table = np.ascontiguousarray(load_item_emb(split))
    cemb = np.load(FLAT / split / "cand_emb_idx.npy").astype(np.int64)
    hemb = np.load(FLAT / split / "hist_emb_idx.npy").astype(np.int64)
    het = np.load(FLAT / split / "hist_event_type.npy")
    for v in variants:
        Kv, decay, ev, cols = VARIANTS[v]
        hmask = np.ones(len(hemb), np.bool_) if ev == 0 else (het == ev)
        out = np.empty((len(cemb), len(NAMES)), np.float32)
        _hn_kernel(co, M, ho, H, cemb, hemb, hmask, table, Kv, decay, out)
        for i in cols:
            nm = NAMES[i].replace("ctx_hn_", f"ctx_hn{v}_") if v and v.startswith("_") else NAMES[i]
            save(split, nm, out[:, i])
        print(split, "hn", v, f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    vs = ["" if v == "base" else v for v in sys.argv[2].split(",")]
    for s in sys.argv[1].split(","):
        build(s, vs)
