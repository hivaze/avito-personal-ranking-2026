"""Information for the pop1 design: NOISY-HIGH / RELIABLE-HIGH act/exp of the existing 2-epoch screen runs (abl_base vs abl_xdrop_pop,
seeds 23/24, LAST checkpoints = the frozen primary read), 2-member rz-mean pools. Same cell / act-exp definitions as f04."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/followup")
from fu_common import *
V = Split("val"); y, fid, n = V.y, V.fid, V.n
PA = ROOT / "nn_x/abl/preds"
f5 = np.load(FLAT / "val/cand_f5.npy").astype(np.float64)
nexp = np.nan_to_num(np.load(FEATS / "val/tte_it_imp.npy").astype(np.float64)) + np.nan_to_num(np.load(FEATS / "val/it_hist_all_luo.npy").astype(np.float64))
q90 = np.quantile(f5, 0.9)
C = {"NOISY-HIGH": (f5 >= q90) & (nexp < 5), "RELIABLE-HIGH": (f5 >= q90) & (nexp >= 5)}
def expected(s):
    o = np.lexsort((s, fid)); fs_ = np.r_[True, fid[o][1:] != fid[o][:-1]]
    rk = np.arange(len(o)) - np.flatnonzero(fs_)[np.cumsum(fs_) - 1]
    pct = np.empty(len(o)); pct[o] = rk / np.maximum(V.M[fid[o]] - 1, 1)
    dec = np.clip((pct * 10).astype(int), 0, 9)
    rate = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10); ex = rate[dec]
    return ex * (np.bincount(fid, y, minlength=n) / np.maximum(np.bincount(fid, ex, minlength=n), 1e-12))[fid]
G = V.cl.max() + 1; BW = np.random.default_rng(0).multinomial(G, np.full(G, 1.0 / G), size=200).astype(np.float64)
S = {arm: V.rz(np.mean([V.rz(np.load(PA / f"nn_abl_{arm}_s{s}@last_trainonly_val.npy")) for s in (23, 24)], 0)) for arm in ("base", "xdrop_pop")}
EX = {k: expected(s) for k, s in S.items()}
for cn, m in C.items():
    ycl = np.bincount(V.cl[fid], y * m, G); out = {}
    for k in S:
        ecl = np.bincount(V.cl[fid], EX[k] * m, G); out[k] = (ycl.sum() / ecl.sum(), (BW @ ycl) / (BW @ ecl))
        _, se = ratio_cl(np.bincount(fid, y * m, n), np.bincount(fid, EX[k] * m, n), V.cl)
        print(f"2-epoch @last pool {k:10s} NDCG {V.nd(S[k]).mean():.5f} | {cn:13s} act/exp {out[k][0]:.3f} +- {se:.3f}")
    d = out["xdrop_pop"][1] - out["base"][1]
    print(f"   {cn}: xdrop_pop - base {out['xdrop_pop'][0] - out['base'][0]:+.3f} +- {d.std():.3f} (paired cluster bootstrap)")
