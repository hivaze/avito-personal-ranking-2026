"""trd_tp_: population category-transition prior.

History is newest-first, so for events j (newer) and j+L (older) in one feed we observe "category a, then L steps later b".
From ALL train histories (dense population, unlabeled) count T[L, a, b] = #(cat[j+L]=a, cat[j]=b, event_type[j]=1), L=1..LMAX.
PMI_L(a, b) = log P(b | a, L) - log P(b),  P(b) = marginal of b over et1 events (Dirichlet smoothing 1/54 * 20).
The candidate is the next event after the history, i.e. lag j+1 from the event at hpos j:
  trd_tp_l1   = PMI_1(cat[hpos0], c)
  trd_tp_m5   = mean_{j<5}  PMI_{j+1}(cat[hpos j], c)
  trd_tp_m20  = mean_{j<20} PMI_{min(j+1,LMAX)}(cat[hpos j], c)
  trd_tp_x5   = mean over j<5 with cat[hpos j] != c of PMI (cross-category only; NaN if none)
  trd_tp_mx20 = max_{j<20} PMI (cross-category only)
"""
import sys, time
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *

LMAX = 20


@nb.njit(parallel=True, cache=True)
def count_T(hist_off, H, hcat, het, nth, out):
    n = len(H)
    chunk = (n + nth - 1) // nth
    for t in nb.prange(nth):
        for f in range(t * chunk, min(n, (t + 1) * chunk)):
            ho = hist_off[f]; h = H[f]
            for j in range(h):
                if het[ho + j] != 1:
                    continue
                b = hcat[ho + j]
                for L in range(1, LMAX + 1):
                    if j + L >= h:
                        break
                    out[t, L, hcat[ho + j + L], b] += 1


@nb.njit(parallel=True, cache=True)
def cand_feats(cand_off, M, hist_off, H, ccat, hcat, pmi, out):
    n = len(M)
    for f in nb.prange(n):
        ho = hist_off[f]; h = H[f]; co = cand_off[f]
        for i in range(M[f]):
            c = ccat[co + i]
            if h == 0:
                for q in range(5):
                    out[co + i, q] = np.nan
                continue
            out[co + i, 0] = pmi[1, hcat[ho], c]
            s5 = 0.0; n5 = 0; s20 = 0.0; n20 = 0; sx = 0.0; nx = 0; mx = -1e9
            for j in range(min(h, 20)):
                L = min(j + 1, LMAX)
                v = pmi[L, hcat[ho + j], c]
                if j < 5:
                    s5 += v; n5 += 1
                s20 += v; n20 += 1
                if hcat[ho + j] != c:
                    if j < 5:
                        sx += v; nx += 1
                    if v > mx:
                        mx = v
            out[co + i, 1] = s5 / n5
            out[co + i, 2] = s20 / n20
            out[co + i, 3] = sx / nx if nx > 0 else np.nan
            out[co + i, 4] = mx if mx > -1e8 else np.nan


def pmi_table(threads):
    p = CACHE / "tp_pmi.npy"
    if p.exists():
        return np.load(p)
    f = feeds_df("train")
    hc = np.load(FLAT / "train/hist_category_id.npy").astype(np.int64)
    he = np.load(FLAT / "train/hist_event_type.npy")
    T = np.zeros((threads, LMAX + 1, N_CAT, N_CAT), np.float64)
    count_T(f["hist_off"].to_numpy(), f["H"].to_numpy().astype(np.int64), hc, he, threads, T)
    T = T.sum(0)
    pb = T[1].sum(0) + 1.0; pb /= pb.sum()
    pmi = np.zeros_like(T)
    for L in range(1, LMAX + 1):
        cond = (T[L] + 20 * pb[None]) / (T[L].sum(1, keepdims=True) + 20)
        pmi[L] = np.log(cond / pb[None])
    np.save(p, pmi)
    return pmi


def build(split, threads=16):
    nb.set_num_threads(threads)
    t0 = time.time()
    pmi = pmi_table(threads)
    f = feeds_df(split)
    hc = np.load(FLAT / split / "hist_category_id.npy").astype(np.int64)
    cc = np.load(FLAT / split / "cand_category_id.npy").astype(np.int64)
    out = np.empty((len(cc), 5), np.float32)
    cand_feats(f["cand_off"].to_numpy(), f["M"].to_numpy().astype(np.int64), f["hist_off"].to_numpy(),
               f["H"].to_numpy().astype(np.int64), cc, hc, pmi, out)
    for q, nm in enumerate(["l1", "m5", "m20", "x5", "mx20"]):
        np.save(FEATS / split / f"trd_tp_{nm}.npy", out[:, q])
    print(split, "tp done", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1].split(","):
        build(s, int(sys.argv[2]) if len(sys.argv) > 2 else 16)
