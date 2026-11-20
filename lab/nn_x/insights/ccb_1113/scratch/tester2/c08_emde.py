"""C08 EMDE multi-partition sketch (PREDECLARED_ccb.txt §C08; own history only, label-free, B10).
32 partitions x 6 Gaussian projections (seed 0) of the L2-normalised int8 item vector, each projection cut at its median
(quantile fitted on a 5M-row label-free uniform sample of history + candidate rows of train/val/test) -> 2^6 = 64 buckets.
Per feed and partition n: S_n = exp(-hpos/64)-weighted, L1-normalised bucket histogram of the history events with a vector.
emde_all_k64 = mean_n log(1e-4 + S_n[b_n(c)]); emde_e1 / emde_e4 / emde_c23 (types 2,3) / emde_dlv (types 5,6,8) = same over
those event types (NaN if the group is empty or the candidate has no vector);
emde_all_k64_lift = emde_all_k64 - mean_n log P_global(b_n(c)) (P_global = bucket shares in the 5M sample). 6 columns.
"""
import sys
sys.path.insert(0, "/tmp/scratchpad/ccb/tester2")
from harness import *  # noqa: F401,F403
from emb_util import load_side, emb_memmap

NAMES = ["emde_all_k64", "emde_e1", "emde_e4", "emde_c23", "emde_dlv", "emde_all_k64_lift"]
NP, NB = 32, 6
rng = np.random.default_rng(0)
R = rng.standard_normal((128, NP * NB)).astype(np.float32)


def sample_cuts():
    """5M uniform rows over hist + cand rows of train/val/test (label-free) -> per-projection medians + bucket shares."""
    t = time.time()
    parts = []
    for sp in ("train", "val", "test"):
        for nm in ("hist_emb_idx", "cand_emb_idx"):
            parts.append((sp, nm, len(np.load(FLAT / sp / f"{nm}.npy", mmap_mode="r"))))
    tot = sum(p[2] for p in parts)
    pos = np.sort(np.random.default_rng(0).choice(tot, 5_000_000, replace=False))
    vecs = []; start = 0
    for sp, nm, n in parts:
        q = pos[(pos >= start) & (pos < start + n)] - start; start += n
        e = np.asarray(np.load(FLAT / sp / f"{nm}.npy", mmap_mode="r")[q]); e = np.sort(e[e >= 0])
        if len(e) == 0:
            continue
        mm = emb_memmap(sp)
        for s in range(0, len(e), 1_000_000):
            vecs.append(mm[e[s:s + 1_000_000]].astype(np.float32))
    X = np.concatenate(vecs); nrm = np.linalg.norm(X, axis=1, keepdims=True); X /= np.where(nrm > 0, nrm, 1.0)
    P = X @ R
    cuts = np.median(P, axis=0).astype(np.float32)
    codes = to_codes(P, cuts)
    pg = np.stack([np.bincount(codes[:, n], minlength=64) for n in range(NP)]).astype(np.float64)
    pg = (pg + 1.0) / (pg.sum(1, keepdims=True) + 64.0)
    log(f"C08 sample: {len(X)} vectors with emb (of 5M sampled rows over {tot} rows), cuts fitted ({time.time()-t:.0f}s); "
        f"P_global min {pg.min():.5f} max {pg.max():.5f}")
    return cuts, pg


def to_codes(P, cuts):
    bits = (P > cuts).reshape(len(P), NP, NB).astype(np.uint8)
    return (bits * (1 << np.arange(NB, dtype=np.uint8))[None, None, :]).sum(2).astype(np.uint8)


@nb.njit(parallel=True, cache=True)
def kern(H, hoff, M, coff, h_loc, h_type, h_pos, c_loc, codes, lpg, out):
    NPn = codes.shape[1]
    for f in nb.prange(len(M)):
        h = H[f]; ho = hoff[f]; co = coff[f]
        S = np.zeros((5, NPn, 64)); tot = np.zeros(5)
        for j in range(h):
            q = ho + j; hl = h_loc[q]
            if hl < 0:
                continue
            w = np.exp(-h_pos[q] / 64.0); ty = h_type[q]
            g0 = 0; g1 = ty == 1; g2 = ty == 4; g3 = (ty == 2) or (ty == 3); g4 = (ty == 5) or (ty == 6) or (ty == 8)
            for g in range(5):
                use = (g == 0) or (g == 1 and g1) or (g == 2 and g2) or (g == 3 and g3) or (g == 4 and g4)
                if use:
                    tot[g] += w
                    for n in range(NPn):
                        S[g, n, codes[hl, n]] += w
        for i in range(M[f]):
            r = co + i; cl = c_loc[r]
            for g in range(5):
                if cl < 0 or tot[g] <= 0:
                    out[r, g] = np.nan
                    continue
                acc = 0.0
                for n in range(NPn):
                    acc += np.log(1e-4 + S[g, n, codes[cl, n]] / tot[g])
                out[r, g] = acc / NPn
            if cl < 0 or tot[0] <= 0:
                out[r, 5] = np.nan
            else:
                acc = 0.0
                for n in range(NPn):
                    acc += lpg[n, codes[cl, n]]
                out[r, 5] = out[r, 0] - acc / NPn


def build(split, sel, cuts, lpg):
    t = time.time()
    s = load_side(split, sel)
    codes = to_codes(s["V"] @ R, cuts)
    out = np.zeros((len(s["c_loc"]), len(NAMES)), np.float64)
    kern(s["H"], s["hoff"], s["M"], s["coff"], s["h_loc"], s["h_type"], s["h_pos"], s["c_loc"], codes, lpg, out)
    log(f"C08 {split}: {len(s['M'])} feeds, {len(out)} rows built ({time.time()-t:.0f}s)")
    return out


if __name__ == "__main__":
    nb.set_num_threads(12)
    LOG = open(HERE / "logs/c08.log", "a")
    log = log_to(LOG)
    log("==== C08 EMDE sketch", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    sanity(log)
    cuts, pg = sample_cuts()
    lpg = np.log(pg)
    Xv = build("val", None, cuts, lpg); assert Xv.shape[0] == len(yv)
    Xh = build("train", HOLD, cuts, lpg); assert Xh.shape[0] == len(yh)
    np.save(FEATD / "c08_val.npy", Xv.astype(np.float32)); np.save(FEATD / "c08_hold.npy", Xh.astype(np.float32))
    res = evaluate("C08", NAMES, [Xv[:, i] for i in range(len(NAMES))], [Xh[:, i] for i in range(len(NAMES))], log)
    with open(HERE / "logs/results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    log("DONE")
