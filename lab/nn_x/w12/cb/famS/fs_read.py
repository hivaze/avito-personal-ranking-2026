"""famS reads (evaluation only; val / holdout labels are used here and nowhere else).
  python fs_read.py nats  <tag> [<tag> ...]            one nats.py line per tag
  python fs_read.py bundle <name> <tag> <tag> [...]     nats.py line of the member-equal z-mean bundle
  python fs_read.py cmp   <tagA> <tagB>                 paired per-feed NDCG delta A - B on val and holdout
  python fs_read.py mbin  <tag> [...]                   level and gap to W16 by M bin (val: M < 50 / >= 50; holdout: all M >= 50)
  python fs_read.py joint <tagS> <tagH>                 conditional logit [W16], [W16,S], [W16,H], [W16,S,H], [W16, bundle(S,H)]
Tags resolve to famS/preds/<tag>_{val,hold}.npy, then famS/staged/, then infra/out/."""
import sys, json
import numpy as np
sys.path.insert(0, '<repo>/nn_x/w12/cb')
import nats
from nats import screen, clogit, _meta, ndcg_fast, se
from pathlib import Path
R = Path('<repo>/nn_x/w12/cb')


def P(tag):
    for d in (R / 'famS/preds', R / 'famS/staged', R / 'infra/out'):
        if (d / f'{tag}_val.npy').exists():
            return np.load(d / f'{tag}_val.npy'), np.load(d / f'{tag}_hold.npy')
    raise FileNotFoundError(tag)


def nd(sp, s):
    m = _meta()
    return ndcg_fast(np.asarray(s, np.float32), m.y[sp], m.off[sp], m.M[sp])


def cmp(a, b, verbose=True):
    (av, ah), (bv, bh) = P(a), P(b)
    dv = nd('val', av) - nd('val', bv); dh = nd('hold', ah) - nd('hold', bh)
    zp = (dv.mean() / se(dv) + dh.mean() / se(dh)) / np.sqrt(2)
    r = dict(a=a, b=b, val=float(dv.mean()), val_se=float(se(dv)), hold=float(dh.mean()), hold_se=float(se(dh)), pooled_z=float(zp), n_val=len(dv), n_hold=len(dh))
    if verbose:
        print(f"[{a} - {b}] val {r['val']:+.5f} +- {r['val_se']:.5f} (n={len(dv)}) | holdout {r['hold']:+.5f} +- {r['hold_se']:.5f} (n={len(dh)}) | pooled z {zp:+.2f}")
    return r


def mbin(tag):
    m = _meta(); xv, xh = P(tag); out = {}
    for sp, x in (('val', xv), ('hold', xh)):
        e = nd(sp, x); w = nd(sp, m.zw[sp]); M = m.M[sp]
        for nm, k in (('M<50', M < 50), ('M>=50', M >= 50), ('all', M > 0)):
            if k.sum() == 0:
                continue
            g = (w - e)[k]
            out[f'{sp} {nm}'] = dict(n=int(k.sum()), level=float(e[k].mean()), w16=float(w[k].mean()), gap=float(g.mean()), gap_se=float(se(g)))
            print(f"[{tag}] {sp:4s} {nm:6s} n={int(k.sum()):6d} X {e[k].mean():.5f} W16 {w[k].mean():.5f} gap {g.mean():.5f} +- {se(g):.5f} (relative {g.mean() / w[k].mean():.3f})")
    return out


def joint(s, h):
    m = _meta(); out = {}
    for sp, i in (('val', 0), ('hold', 1)):
        zs = m.z(sp, P(s)[i]); zh = m.z(sp, P(h)[i]); zw = m.zw[sp]; zb = m.z(sp, (zs + zh) / 2)
        y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]
        _, _, l0 = clogit(zw[:, None], y, off, M); nf = int((l0 != 0).sum())
        res = {}
        for nm, cols in (('W16+S', [zw, zs]), ('W16+H', [zw, zh]), ('W16+S+H', [zw, zs, zh]), ('W16+bundle(S,H)', [zw, zb])):
            b, sb, l = clogit(np.stack(cols, 1), y, off, M); d = l0 - l
            res[nm] = dict(nats=float(d.sum() / nf), se=float(np.std(d) / np.sqrt(len(d)) * len(d) / nf), beta=[float(v) for v in b], beta_z=[float(v) for v in b / sb])
        # gain of adding H on top of [W16, S] and S on top of [W16, H], paired
        for nm, base, cols in (('H | W16,S', [zw, zs], [zw, zs, zh]), ('S | W16,H', [zw, zh], [zw, zs, zh])):
            _, _, lb = clogit(np.stack(base, 1), y, off, M); _, _, lf = clogit(np.stack(cols, 1), y, off, M); d = lb - lf
            res[nm] = dict(nats=float(d.sum() / nf), se=float(np.std(d) / np.sqrt(len(d)) * len(d) / nf))
        res['corr(S,H)'] = float(np.corrcoef(zs, zh)[0, 1])
        # residual correlation given W16 (partial correlation of the parts not explained by z_W16)
        rs = zs - np.dot(zs, zw) / np.dot(zw, zw) * zw; rh = zh - np.dot(zh, zw) / np.dot(zw, zw) * zw
        res['partial_corr(S,H | W16)'] = float(np.corrcoef(rs, rh)[0, 1])
        out[sp] = res
        print(f"[{sp}] " + " | ".join(f"{k}: {v['nats']:+.5f}+-{v['se']:.5f}" + (f" (beta z {', '.join(f'{t:+.1f}' for t in v['beta_z'])})" if 'beta_z' in v else '') for k, v in res.items() if isinstance(v, dict)))
        print(f"[{sp}] corr(S,H) {res['corr(S,H)']:.4f}; partial corr given W16 {res['partial_corr(S,H | W16)']:.4f}")
    return out


if __name__ == '__main__':
    c = sys.argv[1]; a = sys.argv[2:]
    if c == 'nats':
        for t in a:
            xv, xh = P(t); r = screen(xv, xh, tag=t)
            (R / 'famS/reads').mkdir(exist_ok=True); (R / f'famS/reads/nats_{t}.json').write_text(json.dumps(r))
    elif c == 'bundle':
        ps = [P(t) for t in a[1:]]; r = screen([p[0] for p in ps], [p[1] for p in ps], tag=a[0]); r['members'] = a[1:]
        (R / 'famS/reads').mkdir(exist_ok=True); (R / f'famS/reads/nats_{a[0]}.json').write_text(json.dumps(r))
    elif c == 'cmp':
        r = cmp(a[0], a[1]); (R / 'famS/reads').mkdir(exist_ok=True); (R / f'famS/reads/cmp_{a[0]}__{a[1]}.json').write_text(json.dumps(r))
    elif c == 'mbin':
        for t in a:
            r = mbin(t); (R / 'famS/reads').mkdir(exist_ok=True); (R / f'famS/reads/mbin_{t}.json').write_text(json.dumps(r))
    elif c == 'joint':
        r = joint(a[0], a[1]); (R / 'famS/reads').mkdir(exist_ok=True); (R / f'famS/reads/joint_{a[0]}__{a[1]}.json').write_text(json.dumps(r))
