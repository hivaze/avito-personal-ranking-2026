"""Paired per-feed difference NATS(X) - NATS(H): gain of [z_W16, z_X] over [z_W16] minus gain of [z_W16, z_H] over [z_W16], same feeds (evaluation only;
conditional logit of nats.py, coefficients fitted on the witness itself). Third run of the famS task; reproduces reads/natsdiff_vs_hard.json of the second run.
usage: python fs_natsdiff.py <H tag> <X tag | tagA+tagB+... (member-equal z-mean bundle)> [...]  -> reads/natsdiff3_vs_<H>.json"""
import sys, json
import numpy as np
sys.path.insert(0, '<repo>/nn_x/w12/cb'); sys.path.insert(0, '<repo>/nn_x/w12/cb/famS')
from nats import clogit, _meta
from fs_read import P, R
m = _meta(); H = sys.argv[1]; out = {}


def gain(sp, i, name):
    zx = np.mean([m.z(sp, np.asarray(P(t)[i], np.float32)) for t in name.split('+')], 0); zw = m.zw[sp]
    y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]
    _, _, lw = clogit(zw[:, None], y, off, M); _, _, l2 = clogit(np.stack([zw, zx], 1), y, off, M)
    return lw - l2, int((lw != 0).sum())


gh = {sp: gain(sp, i, H) for sp, i in (('val', 0), ('hold', 1))}
for t in sys.argv[2:]:
    r = {}
    for sp, i in (('val', 0), ('hold', 1)):
        gx, nf = gain(sp, i, t); d = gx - gh[sp][0]
        r[sp] = dict(diff=float(d.sum() / nf), se=float(np.std(d) / np.sqrt(len(d)) * len(d) / nf))
    r['pooled_z'] = float((r['val']['diff'] / r['val']['se'] + r['hold']['diff'] / r['hold']['se']) / np.sqrt(2)); out[t] = r
    print(f"[NATS({t}) - NATS({H})] val {r['val']['diff']:+.5f} +- {r['val']['se']:.5f} | holdout {r['hold']['diff']:+.5f} +- {r['hold']['se']:.5f} | pooled z {r['pooled_z']:+.2f}")
(R / f'famS/reads/natsdiff3_vs_{H}.json').write_text(json.dumps(out))
