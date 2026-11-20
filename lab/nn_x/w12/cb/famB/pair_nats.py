"""Paired comparison of two models or bundles: NATS(A) - NATS(B) per feed and level(A) - level(B), on val and holdout.
usage: python -u pair_nats.py A B [A2 B2 ...]     A / B = tag or tag+tag+... (bundle, member-equal z-mean); "b0" = infra ts_lr020_s0"""
import sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB"); sys.path.insert(0, "<repo>/nn_x/w12/cb")
from fb_common import *
import nats
from nats import clogit
m = nats._meta()
P = lambda tag, sp: np.load((B0V if sp == "val" else B0H) if tag == "b0" else PREDS / f"{tag}_{sp}.npy")
def zb(b, sp):
    return np.mean([m.z(sp, P(t, sp)) for t in b.split("+")], 0)
def stats(b, sp):
    y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]; zw = m.zw[sp]; zx = zb(b, sp)
    _, _, lw = clogit(zw[:, None], y, off, M); _, _, l2 = clogit(np.stack([zw, zx], 1), y, off, M)
    return lw - l2, int((lw != 0).sum()), nats.ndcg_fast(zx.astype(np.float32), m.y[sp], off, M)
a = sys.argv[1:]
for A, B in zip(a[0::2], a[1::2]):
    out = []
    for sp in ("val", "hold"):
        da, nf, na = stats(A, sp); db, _, nb_ = stats(B, sp); d = da - db; dl = na - nb_
        out.append(f"{sp}: NATS {da.sum() / nf:+.5f} vs {db.sum() / nf:+.5f}, diff {d.sum() / nf:+.5f}+-{np.std(d) / np.sqrt(len(d)) * len(d) / nf:.5f} | level {na.mean():.5f} vs {nb_.mean():.5f}, diff {dl.mean():+.5f}+-{dl.std() / np.sqrt(len(dl)):.5f}")
    print(f"[{A}] - [{B}]\n   " + "\n   ".join(out), flush=True)
