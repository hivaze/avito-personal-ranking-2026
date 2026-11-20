"""famS read by M bin (lead's notice of 02:19Z, point 3): nats and member-rule statistic on val M >= 50 (and val M < 50) separately.
Evaluation only. The conditional logit is refitted on the subset of feeds (same convention as nats.py: coefficients fitted on the witness itself).
  python fs_m50.py <tag | tagA+tagB+... (bundle, member-equal z-mean)> [...]
Writes reads/m50_<name>.json."""
import sys, json
import numpy as np
sys.path.insert(0, '<repo>/nn_x/w12/cb')
sys.path.insert(0, '<repo>/nn_x/w12/cb/famS')
from nats import clogit, _meta, ndcg_fast, se
from fs_read import P, R


def read(name):
    m = _meta(); tags = name.split('+'); out = {'tag': name}
    xv = [P(t)[0] for t in tags]
    zx = np.mean([m.z('val', np.asarray(a, np.float32)) for a in xv], 0); zw = m.zw['val']
    y, off, M = m.y['val'], m.off['val'], m.M['val']
    e0 = ndcg_fast(zw, y, off, M); ex = ndcg_fast(zx, y, off, M)
    dm = ndcg_fast(0.9 * m.z('val', zw) + 0.1 * m.z('val', zx), y, off, M) - e0
    for nm, k in (('M<50', M < 50), ('M>=50', M >= 50)):
        rows = np.repeat(k, M); Ms = M[k]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)
        ys = y[rows].astype(np.float64)
        _, _, lw = clogit(zw[rows][:, None], ys, offs, Ms); b2, s2, l2 = clogit(np.stack([zw[rows], zx[rows]], 1), ys, offs, Ms)
        d = lw - l2; nf = int((lw != 0).sum())
        r = dict(n=int(k.sum()), n_with_click=nf, nats=float(d.sum() / nf), nats_se=float(np.std(d) / np.sqrt(len(d)) * len(d) / nf), beta_x_z=float(b2[1] / s2[1]),
                 level=float(ex[k].mean()), w16=float(e0[k].mean()), gap=float((e0 - ex)[k].mean()), gap_se=float(se((e0 - ex)[k])),
                 mix01=float(dm[k].mean()), mix01_se=float(se(dm[k])))
        out[nm] = r
        print(f"[{name}] val {nm:6s} n={r['n']:5d} level {r['level']:.5f} (W16 {r['w16']:.5f}, gap {r['gap']:.5f}+-{r['gap_se']:.5f}) | NATS {r['nats']:+.5f}+-{r['nats_se']:.5f} (beta_X z {r['beta_x_z']:+.1f})"
              f" | member rule 0.9/0.1 {r['mix01']:+.5f}+-{r['mix01_se']:.5f} (z {r['mix01'] / r['mix01_se']:+.2f})")
    return out


if __name__ == '__main__':
    for t in sys.argv[1:]:
        r = read(t); (R / f'famS/reads/m50_{t}.json').write_text(json.dumps(r))
