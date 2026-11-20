"""critique2 N4: recompute the wave-5 stage-E TRUNC readout (the number that closed TRUNC: pooled member val -0.00048 +- 0.00051,
holdout -0.00009 +- 0.00029; 4-member pool val -0.00034 / holdout -0.00087 +- 0.00038) from the stored prediction files in
nn_x/w5/preds (selected checkpoints and @last), with per-seed deltas, and the same for the 4-member pools, plus the pool
added to G3 (T1 = G3 + 4 trunc members vs G3 + 4 base members) which the chain never computed because it stopped on FAIL.
CPU only. Writes out/c2_05_trunc_e.log (stdout)."""
import os, glob
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
import sys
from pathlib import Path
import numpy as np
ROOT = Path("<repo>"); sys.path.insert(0, str(ROOT / "src"))
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
W5 = ROOT / "nn_x/w5/preds"
fv, Mv, ov = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
hm = np.load(ROOT / "preds/holdout/holdout_meta.npy"); Mh = hm[0].astype(np.int64); offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); yh = np.load(ROOT / "preds/holdout/holdout_y.npy")
print("E files:", sorted(os.path.basename(p) for p in glob.glob(str(W5 / "nn_w5_e10_*_trainonly_val.npy"))))
print("E holdout files:", sorted(os.path.basename(p) for p in glob.glob(str(W5 / "holdout/nn_w5_e10_*"))))
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
zh = lambda x: feed_rank_z(offh, Mh, np.asarray(x, np.float32))[:, 1]
def pe(a, b, n=None):
    d = a - b; return f"{d.mean():+.5f} ± {d.std(ddof=1)/np.sqrt(len(d)):.5f}"
for suf in ["", "@last"]:
    dv, dh, zb, zt, hb, ht = [], [], [], [], [], []
    for s in [951, 952, 953, 954]:
        pb, pt = W5 / f"nn_w5_e10_base_s{s}{suf}_trainonly_val.npy", W5 / f"nn_w5_e10_trunc_s{s}{suf}_trainonly_val.npy"
        if not (pb.exists() and pt.exists()):
            print(f"  missing {pb.name} or {pt.name}"); continue
        b, t = np.load(pb), np.load(pt)
        nb, nt = ndcg_per_feed(b, y, ov, Mv), ndcg_per_feed(t, y, ov, Mv); dv.append(nt - nb); zb.append(zv(b)); zt.append(zv(t))
        line = f"  seed {s}{suf or ' (selected)'}: base val {nb.mean():.5f} trunc {nt.mean():.5f} d {pe(nt, nb)}"
        hb_, ht_ = W5 / f"holdout/nn_w5_e10_base_s{s}{suf}_holdout.npy", W5 / f"holdout/nn_w5_e10_trunc_s{s}{suf}_holdout.npy"
        if hb_.exists() and ht_.exists():
            hbb, htt = np.load(hb_), np.load(ht_); nhb, nht = ndcg_per_feed(hbb, yh, offh, Mh), ndcg_per_feed(htt, yh, offh, Mh)
            dh.append(nht - nhb); hb.append(zh(hbb)); ht.append(zh(htt)); line += f" | holdout base {nhb.mean():.5f} trunc {nht.mean():.5f} d {pe(nht, nhb)}"
        print(line)
    if dv:
        D = np.mean(dv, 0); print(f"  POOLED member{suf}: val {D.mean():+.5f} ± {D.std(ddof=1)/np.sqrt(len(D)):.5f} (per-feed SE of the seed-mean delta; seed-level sd of the 4 deltas {np.std([d.mean() for d in dv], ddof=1):.5f})")
        pb_, pt_ = ndcg_per_feed(np.mean(zb, 0), y, ov, Mv), ndcg_per_feed(np.mean(zt, 0), y, ov, Mv)
        print(f"  4-member POOL{suf}: base val {pb_.mean():.5f} trunc {pt_.mean():.5f} d {pe(pt_, pb_)}")
    if dh:
        D = np.mean(dh, 0); print(f"  POOLED member{suf}: holdout {D.mean():+.5f} ± {D.std(ddof=1)/np.sqrt(len(D)):.5f}")
        pb_, pt_ = ndcg_per_feed(np.mean(hb, 0), yh, offh, Mh), ndcg_per_feed(np.mean(ht, 0), yh, offh, Mh)
        print(f"  4-member POOL{suf}: base holdout {pb_.mean():.5f} trunc {pt_.mean():.5f} d {pe(pt_, pb_)}")
    # added to G3 (TO branch only, val): G3_to (20 members) + 4 base vs + 4 trunc, member-equal z-mean
    if zb:
        g3to = np.load(ROOT / "preds/ens_G3_to_val.npy"); zg = zv(g3to)
        A = ndcg_per_feed((20 * zg + np.sum(zb, 0)) / 24, y, ov, Mv); B = ndcg_per_feed((20 * zg + np.sum(zt, 0)) / 24, y, ov, Mv); G = ndcg_per_feed(zg, y, ov, Mv)
        print(f"  G3-TO ({G.mean():.5f}) + 4 base: {A.mean():.5f} ({pe(A, G)}) | + 4 trunc: {B.mean():.5f} ({pe(B, G)}) | trunc - base inside G3: {pe(B, A)}")
print("done")
