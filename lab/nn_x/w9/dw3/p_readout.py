"""INFORMATION read-out of the WAVE-9 arm DW3 (never a gate; nn_x/w5/logs/E_dw3_predeclared.txt): the idea's (P) swap pool,
cold/warm-positive val slices and the member correlation matrix. CPU only, read-only.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/dw3/p_readout.py E   (stage E: nn_x/w5/preds,
        w5_e10_dw3_s951/952 vs w5_e10_base_s951-954, selected ckpt)   -> nn_x/w9/dw3/logs/p_readout_E.txt (+ a copy nn_x/w5/logs/p_readout_dw3.txt)
  ... p_readout.py S   (stage S: nn_x/w9/dw3/preds, w9_dw3_s23/24 vs w9_base_s23/24, LAST ckpt; cold/warm slices + corr only) -> logs/p_readout_S.txt
(P) swap pool: z-mean{dw3_951, dw3_952, base_953, base_954} - z-mean{base_951..954} on val and holdout (the idea's bar: >= +0.0003 val AND >= 0
holdout). coldpos / warmpos = val feeds whose positives are all cold / all warm (it_hist_all_luo == 0 = cold; the r6_eval.py CPU2 definition).
"""
import os
import sys
import time

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
sys.path.insert(0, "<repo>/src")
from common import *  # noqa: E402,F401,F403
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402

ST = sys.argv[1] if len(sys.argv) > 1 else "E"
D = Path("<repo>/nn_x/w9/dw3")
if ST == "E":
    W = Path("<repo>/nn_x/w5/preds")
    ARM, BASE, SUF = [f"w5_e10_dw3_s{s}" for s in (951, 952)], [f"w5_e10_base_s{s}" for s in (951, 952, 953, 954)], ""
else:
    W = D / "preds"
    ARM, BASE, SUF = [f"w9_dw3_s{s}" for s in (23, 24)], [f"w9_base_s{s}" for s in (23, 24)], "@last"
L = []
P = lambda s: (print(s, flush=True), L.append(s))  # noqa: E731
fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(W / "holdout/holdout_meta.npy")
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
yh = np.load(W / "holdout/holdout_y.npy")
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
fid = np.repeat(np.arange(len(Mv)), Mv)
luo = np.load("<repo>/nn_x/r6/feat_cache/val/it_hist_all_luo.npy", mmap_mode="r")
cold = np.nan_to_num(np.asarray(luo, dtype=np.float64), nan=0.0) == 0
npos, ncold = np.bincount(fid, yv.astype(np.float64), len(Mv)), np.bincount(fid, yv * cold, len(Mv))
SL = {"coldpos": (npos > 0) & (ncold == npos), "warmpos": (npos > 0) & (ncold == 0), "mixedpos": (npos > 0) & (ncold > 0) & (ncold < npos)}
P(f"== DW3 INFORMATION read-out stage {ST} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}): {ARM} vs {BASE} ({'selected' if not SUF else 'LAST'} ckpt)")
ZV, ZH = {}, {}
for v in ARM + BASE:
    ZV[v] = zv(np.load(W / f"nn_{v}{SUF}_trainonly_val.npy"))
    ZH[v] = zh(np.load(W / f"holdout/nn_{v}{SUF}_holdout.npy"))
# cold / warm-positive slices of the paired member delta
dv = np.mean([ev(ZV[a]) - ev(ZV[b]) for a, b in zip(ARM, BASE)], 0)
P("member val slices: " + " | ".join(f"{n} ({m.sum()} feeds) {dv[m].mean():+.5f} (z {dv[m].mean() / max(se(dv[m]), 1e-9):+.1f})" for n, m in SL.items())
  + f" | all {dv.mean():+.5f}")
# member correlation matrix (val rank-z)
names = ARM + BASE
P("member corr (val rank-z): " + "; ".join(f"{a[-4:]}x{b[-4:]} {corr(ZV[a], ZV[b]):.4f}" for i, a in enumerate(names) for b in names[i + 1:]))
xc = [corr(ZV[a], ZV[b]) for a in ARM for b in BASE if a[-4:] != b[-4:]]
bc = [corr(ZV[a], ZV[b]) for i, a in enumerate(BASE) for b in BASE[i + 1:]]
P(f"mean cross-seed arm-base corr {np.mean(xc):.4f} (target <= 0.92) | mean base-base corr {np.mean(bc):.4f} | arm-arm {corr(ZV[ARM[0]], ZV[ARM[1]]):.4f}")
if ST == "E":
    swap = [ARM[0], ARM[1], BASE[2], BASE[3]]
    pv = ev(zv(np.mean([ZV[v] for v in swap], 0))) - ev(zv(np.mean([ZV[v] for v in BASE], 0)))
    ph = eh(zh(np.mean([ZH[v] for v in swap], 0))) - eh(zh(np.mean([ZH[v] for v in BASE], 0)))
    ok = pv.mean() >= 0.0003 and ph.mean() >= 0
    P(f"(P) swap pool z-mean{{dw3_951, dw3_952, base_953, base_954}} - z-mean{{base_951..954}}: val {pv.mean():+.5f}+-{se(pv):.5f} | holdout {ph.mean():+.5f}+-{se(ph):.5f}"
      f" -> idea's (P) bar (>= +0.0003 val AND >= 0 holdout): {'met' if ok else 'not met'} (information)")
    pv4 = ev(zv(np.mean([ZV[v] for v in ARM + BASE[2:]], 0))) - ev(zv(np.mean([ZV[v] for v in BASE[:2]], 0)))
    P(f"(info) 4-member {{dw3 x2 + base 953/954}} vs 2-member base{{951,952}}: val {pv4.mean():+.5f}")
out = D / "logs" / f"p_readout_{ST}.txt"
open(out, "w").write("\n".join(L) + "\n")
if ST == "E":
    open("<repo>/nn_x/w5/logs/p_readout_dw3.txt", "w").write("\n".join(L) + "\n")
print(f"wrote {out}")
