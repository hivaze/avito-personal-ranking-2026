"""C02 Streak / EWMA + per-type same-category recency (PREDECLARED_ccb.txt §C02; own history only, label-free, B10).
S = cos(candidate, history item) on L2-normalised int8 vectors; k = hpos (0 = newest); w_k = exp(-k/tau).
dmean_tau, dlse_tau over history events WITH a vector (candidate must have a vector); dcat_tau over all history events;
tau in {4,16,64}; empty -> NaN. dlse16_t1s1 / dcat16_t1s1: tau 16 over event_type 1 & event_source 1 events.
strike = sum_{k<16} 2^-k 1[cat_h(k) = cat_c]; rec_e1/rec_e4/rec_e23 = -log1p(min hpos of a same-category event of type 1 / 4 /
2-or-3), NaN if none. 15 columns.
"""
import sys
sys.path.insert(0, "/tmp/scratchpad/ccb/tester2")
from harness import *  # noqa: F401,F403
from emb_util import load_side

NAMES = ["dmean4", "dmean16", "dmean64", "dlse4", "dlse16", "dlse64", "dcat4", "dcat16", "dcat64",
         "dlse16_t1s1", "dcat16_t1s1", "strike", "rec_e1", "rec_e4", "rec_e23"]


@nb.njit(parallel=True, cache=True)
def kern(H, hoff, M, coff, h_loc, h_cat, h_type, h_src, h_pos, c_loc, c_cat, V, out):
    taus = np.array([4.0, 16.0, 64.0])
    D = V.shape[1]
    for f in nb.prange(len(M)):
        h = H[f]; ho = hoff[f]; co = coff[f]
        for i in range(M[f]):
            r = co + i
            cl = c_loc[r]; cc = c_cat[r]
            sw = np.zeros(3); sws = np.zeros(3); swe = np.zeros(3); swa = np.zeros(3); swc = np.zeros(3)
            t_sw = 0.0; t_swe = 0.0; t_swa = 0.0; t_swc = 0.0
            strike = 0.0; m1 = 1 << 30; m4 = 1 << 30; m23 = 1 << 30
            for j in range(h):
                q = ho + j
                k = h_pos[q]
                same = h_cat[q] == cc
                ty = h_type[q]
                t1s1 = (ty == 1) and (h_src[q] == 1)
                hl = h_loc[q]
                S = 0.0; valid = False
                if hl >= 0 and cl >= 0:
                    valid = True
                    acc = np.float32(0.0)
                    for d in range(D):
                        acc += V[hl, d] * V[cl, d]
                    S = float(acc)
                e5 = np.exp(5.0 * S)
                for a in range(3):
                    w = np.exp(-k / taus[a])
                    swa[a] += w
                    if same:
                        swc[a] += w
                    if valid:
                        sw[a] += w; sws[a] += w * S; swe[a] += w * e5
                if t1s1:
                    w = np.exp(-k / 16.0)
                    t_swa += w
                    if same:
                        t_swc += w
                    if valid:
                        t_sw += w; t_swe += w * e5
                if same:
                    if k < 16:
                        strike += 2.0 ** (-k)
                    if ty == 1 and k < m1:
                        m1 = k
                    if ty == 4 and k < m4:
                        m4 = k
                    if (ty == 2 or ty == 3) and k < m23:
                        m23 = k
            for a in range(3):
                out[r, a] = sws[a] / sw[a] if sw[a] > 0 else np.nan
                out[r, 3 + a] = 0.2 * np.log(swe[a] / sw[a]) if sw[a] > 0 else np.nan
                out[r, 6 + a] = swc[a] / swa[a] if swa[a] > 0 else np.nan
            out[r, 9] = 0.2 * np.log(t_swe / t_sw) if t_sw > 0 else np.nan
            out[r, 10] = t_swc / t_swa if t_swa > 0 else np.nan
            out[r, 11] = strike
            out[r, 12] = -np.log1p(m1) if m1 < (1 << 30) else np.nan
            out[r, 13] = -np.log1p(m4) if m4 < (1 << 30) else np.nan
            out[r, 14] = -np.log1p(m23) if m23 < (1 << 30) else np.nan


def build(split, sel):
    t = time.time()
    s = load_side(split, sel)
    log(f"C02 {split}: loaded {len(s['M'])} feeds, {len(s['c_cat'])} cand rows, {len(s['h_cat'])} hist rows, "
        f"{len(s['V'])} vectors ({time.time()-t:.0f}s); cand no-vector share {(s['c_loc'] < 0).mean():.4f}")
    out = np.zeros((len(s["c_cat"]), len(NAMES)), np.float64)
    kern(s["H"], s["hoff"], s["M"], s["coff"], s["h_loc"], s["h_cat"], s["h_type"], s["h_src"], s["h_pos"],
         s["c_loc"], s["c_cat"], s["V"], out)
    log(f"C02 {split}: built ({time.time()-t:.0f}s)")
    return out


if __name__ == "__main__":
    nb.set_num_threads(12)
    LOG = open(HERE / "logs/c02.log", "a")
    log = log_to(LOG)
    log("==== C02 streak / EWMA / per-type recency", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    sanity(log)
    Xv = build("val", None)
    assert Xv.shape[0] == len(yv)
    Xh = build("train", HOLD)
    assert Xh.shape[0] == len(yh)
    np.save(FEATD / "c02_val.npy", Xv.astype(np.float32)); np.save(FEATD / "c02_hold.npy", Xh.astype(np.float32))
    res = evaluate("C02", NAMES, [Xv[:, i] for i in range(len(NAMES))], [Xh[:, i] for i in range(len(NAMES))], log)
    with open(HERE / "logs/results.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    log("DONE")
