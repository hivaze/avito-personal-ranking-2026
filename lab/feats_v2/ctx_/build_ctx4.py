"""ctx_hs_: soft competition of feed candidates for the user's recent history items (variants of ctx_hn_soft05).

For the last K non-missing history items h_j (optionally only event_type == e), weight w_j = decay^j:
  mass_i = M * sum_j w_j softmax_i(cos(c_i, h_j) / tau) / sum_j w_j        (1 = uniform share of attention)
  ctx_hs_mutual: recency-weighted count of history items h_j such that c_i is h_j's nearest candidate AND h_j is
                 c_i's nearest history item (mutual nearest neighbours)
Usage: python build_ctx4.py <splits> <variants comma-separated | all> [threads]
"""
import sys, time
import numba as nb
import numpy as np

sys.path.insert(0, "<repo>/feats_v2/ctx_")
from build_ctx import load_meta, load_item_emb, save, FLAT

# name -> (K, decay, tau, event filter (0 = all))
VARIANTS = {
    "ctx_hs_t10": (32, 0.85, 0.10, 0),
    "ctx_hs_k64": (64, 0.90, 0.05, 0),
    "ctx_hs_k128": (128, 0.95, 0.05, 0),
    "ctx_hs_d7": (32, 0.70, 0.05, 0),
    "ctx_hs_e1": (32, 0.85, 0.05, 1),
    "ctx_hs_e2": (32, 0.85, 0.05, 2),
    "ctx_hs_k256": (256, 0.98, 0.05, 0),
    "ctx_hs_e1k64": (64, 0.95, 0.05, 1),
    "ctx_hs_k128t10": (128, 0.95, 0.10, 0),
}


@nb.njit(parallel=True, fastmath=True, cache=True)
def _hs_kernel(cand_off, M, hist_off, H, cemb, hemb, hmask, table, K, decay, tau, out_mass, out_mutual):
    D = table.shape[1]
    for f in nb.prange(len(M)):
        co = cand_off[f]; m = M[f]
        Hm = np.zeros((K, D), np.float32)
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
                out_mass[co + i] = np.nan; out_mutual[co + i] = np.nan
            continue
        S = V @ Hm[:nk].T
        mass = np.zeros(m); mut = np.zeros(m)
        # nearest history item of each candidate
        hbest = np.zeros(m, np.int64)
        for i in range(m):
            b = 0
            for j in range(1, nk):
                if S[i, j] > S[i, b]:
                    b = j
            hbest[i] = b
        wsum = 0.0
        for j in range(nk):
            w = decay ** j
            wsum += w
            b1 = -9.0; ib = -1
            for i in range(m):
                if ok[i] and S[i, j] > b1:
                    b1 = S[i, j]; ib = i
            if ib < 0:
                continue
            z = 0.0
            for i in range(m):
                if ok[i]:
                    z += np.exp((S[i, j] - b1) / tau)
            for i in range(m):
                if ok[i]:
                    mass[i] += w * np.exp((S[i, j] - b1) / tau) / z
            if hbest[ib] == j:
                mut[ib] += w
        for i in range(m):
            if ok[i]:
                out_mass[co + i] = mass[i] / wsum * m
                out_mutual[co + i] = mut[i] / wsum
            else:
                out_mass[co + i] = np.nan; out_mutual[co + i] = np.nan


def build(split, names):
    t = time.time()
    feeds, M, H, co, ho = load_meta(split)
    table = np.ascontiguousarray(load_item_emb(split))
    cemb = np.load(FLAT / split / "cand_emb_idx.npy").astype(np.int64)
    hemb = np.load(FLAT / split / "hist_emb_idx.npy").astype(np.int64)
    het = np.load(FLAT / split / "hist_event_type.npy")
    for nm in names:
        K, decay, tau, ev = VARIANTS[nm]
        hmask = np.ones(len(hemb), np.bool_) if ev == 0 else (het == ev)
        mass = np.empty(len(cemb), np.float32); mut = np.empty(len(cemb), np.float32)
        _hs_kernel(co, M, ho, H, cemb, hemb, hmask, table, K, decay, tau, mass, mut)
        save(split, nm, mass)
        if nm == "ctx_hs_t10":
            save(split, "ctx_hs_mutual", mut)
        print(split, nm, f"{time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    nb.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 24)
    names = list(VARIANTS) if sys.argv[2] == "all" else sys.argv[2].split(",")
    for s in sys.argv[1].split(","):
        build(s, names)
