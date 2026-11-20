"""Stage-E INFORMATION lines of the WAVE-9 HS1 pre-declaration that the two standard read-outs do not print:
  (P) swap pool: z-mean{arm_951, arm_952, base_953, base_954} - z-mean{base_951..954} on val and holdout (hs1.json clause P);
  margin / |member val delta| against the 4:1 removal frontier; member cross-corr vs the 0.92 'material' target (also in div_readout).
CPU only, read-only; reads nn_x/w5/preds after the cp --update=none copy.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w9/hs1/p_swap.py [arm=hs1] [seeds=951,952]
      -> prints + nn_x/w5/logs/p_swap_<arm>.txt
"""
import json
import os
import sys

assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
sys.path.insert(0, "<repo>/src")
from common import *  # noqa: E402,F401,F403
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "hs1"
SEEDS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "951,952").split(",")]
W = Path("<repo>/nn_x/w5/preds")
L = Path("<repo>/nn_x/w5/logs")
fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Mh, oh = np.load(W / "holdout/holdout_meta.npy")
Mh, oh = Mh.astype(np.int64), oh.astype(np.int64)
yh = np.load(W / "holdout/holdout_y.npy")
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)  # noqa: E731
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)  # noqa: E731
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]  # noqa: E731
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]  # noqa: E731
se = lambda d: d.std() / np.sqrt(len(d))  # noqa: E731
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
A = lambda s: f"w5_e10_{ARM}_s{s}"  # noqa: E731
B = lambda s: f"w5_e10_base_s{s}"  # noqa: E731
ZV = {v: zv(np.load(W / f"nn_{v}_trainonly_val.npy")) for v in [A(s) for s in SEEDS] + [B(s) for s in (951, 952, 953, 954)]}
ZH = {v: zh(np.load(W / f"holdout/nn_{v}_holdout.npy")) for v in ZV}
swap_v = zv(np.mean([ZV[A(SEEDS[0])], ZV[A(SEEDS[1])], ZV[B(953)], ZV[B(954)]], 0))
base_v = zv(np.mean([ZV[B(s)] for s in (951, 952, 953, 954)], 0))
swap_h = zh(np.mean([ZH[A(SEEDS[0])], ZH[A(SEEDS[1])], ZH[B(953)], ZH[B(954)]], 0))
base_h = zh(np.mean([ZH[B(s)] for s in (951, 952, 953, 954)], 0))
dpv, dph = ev(swap_v) - ev(base_v), eh(swap_h) - eh(base_h)
dv = np.mean([ev(ZV[A(s)]) - ev(ZV[B(s)]) for s in SEEDS], 0)
rho_x = corr(np.mean([ZV[A(s)] for s in SEEDS], 0), np.mean([ZV[B(s)] for s in (953, 954)], 0))
rho_bb = corr(np.mean([ZV[B(s)] for s in SEEDS], 0), np.mean([ZV[B(s)] for s in (953, 954)], 0))
margin = rho_bb - rho_x
dval = float(dv.mean())
ratio = float("inf") if dval >= 0 else margin / abs(dval)
mcross = float(np.mean([corr(ZV[A(SEEDS[0])], ZV[B(SEEDS[1])]), corr(ZV[A(SEEDS[1])], ZV[B(SEEDS[0])])]))
lines = [f"(P) swap pool {ARM}: z-mean{{{ARM}_{SEEDS[0]}, {ARM}_{SEEDS[1]}, base_953, base_954}} - z-mean{{base_951..954}}: val {dpv.mean():+.5f}+-{se(dpv):.5f} "
         f"(z {dpv.mean() / se(dpv):+.2f}) | holdout {dph.mean():+.5f}+-{se(dph):.5f} (z {dph.mean() / se(dph):+.2f}) -> "
         f"{'OK' if dpv.mean() >= 0.0003 and dph.mean() >= 0 else 'NO'} (P needs val >= +0.0003 AND holdout >= 0; information)",
         f"    4-member pool levels: base {ev(base_v).mean():.5f} / {eh(base_h).mean():.5f}; swap {ev(swap_v).mean():.5f} / {eh(swap_h).mean():.5f}",
         f"frontier: member val delta {dval:+.5f}, pool margin rho_bb - rho_x {margin:+.5f} (rho_bb {rho_bb:.5f}, rho_x {rho_x:.5f}), margin / |dval| "
         f"{ratio:.1f} (4:1 removal frontier; information)",
         f"member cross-corr arm-base (cross-seed) {mcross:.5f} -> {'MATERIAL (<= 0.92)' if mcross <= 0.92 else 'not material (> 0.92 target)'}"]
print("\n".join(lines))
open(L / f"p_swap_{ARM}.txt", "w").write("\n".join(lines) + "\n")
json.dump(dict(arm=ARM, seeds=SEEDS, swap_val=float(dpv.mean()), swap_val_se=float(se(dpv)), swap_hold=float(dph.mean()), swap_hold_se=float(se(dph)),
               dval=dval, margin=margin, ratio=ratio, member_cross=mcross), open(L / f"p_swap_{ARM}.json", "w"), indent=1)
