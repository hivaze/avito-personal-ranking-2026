"""Read-out of E4 (GBDT) and E5 (MLP) second-level arms: val OOF vs reference, vs matched controls, holdout (TO twins),
pmix, slices, seed bagging, near-copy statistics (within-feed Spearman / top-1 agreement with G3 on val OOF and test)."""
from lib import *
import json
import pmix

V = S('val'); HO = S('hold'); TE = S('test')
wpriv = pmix.weights()['private']
log = open(OUT / 'e45_eval.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

REF = {'full': V.ev(np.load(CACHE / 'val_g3.npy')), 'to': V.ev(np.load(CACHE / 'val_pool_to.npy'))}
REFH = HO.ev(np.load(CACHE / 'hold_pool_to.npy'))
g3v = np.load(CACHE / 'val_g3.npy').astype(np.float64); g3t = np.load(CACHE / 'test_g3.npy').astype(np.float64)
pto_v = np.load(CACHE / 'val_pool_to.npy').astype(np.float64); pto_h = np.load(CACHE / 'hold_pool_to.npy').astype(np.float64)

def line(d, cl, w=None, sl=None):
    mu, se = se_cl(d, cl); s = f'{mu:+.5f} +- {se:.5f} (z {mu / se:+.2f})'
    return s, mu, se

def bag(kind, tag, split, sfx):
    fs = [CACHE / f'scr_{kind}_{tag}_s{s}{sfx}_{split}.npy' for s in (1, 2, 3)]
    if not all(f.exists() for f in fs): return None
    sp = {'val': V, 'hold': HO, 'test': TE}[split]
    return sp.z(np.mean([sp.z(np.load(f)) for f in fs], 0).astype(np.float32)).astype(np.float64)

RES = {}
for kind, sfxs in [('gbdt', ['_r100', '_r300']), ('mlp', [''])]:
    for form in ('full', 'to'):
        for arm in ('B0', 'B1', 'B2', 'B3'):
            for sfx in sfxs:
                tag = f'{arm}_{form}'
                fs = [CACHE / f'oof_{kind}_{tag}_s{s}{sfx}.npy' for s in (1, 2, 3)]
                if not all(f.exists() for f in fs):
                    pr(f'{kind} {tag}{sfx}: missing'); continue
                per = [np.load(f) for f in fs]
                ref = REF[form]
                seeds = ' '.join(f'{(p - ref).mean():+.5f}' for p in per)
                dmean = np.mean(per, 0) - ref
                s1, mu, se = line(dmean, V.cl)
                pm, pse = wmean_se(dmean, wpriv)
                # seed-bagged score
                bv = bag(kind, tag, 'val', sfx)
                db = V.ev(bv) - ref; sb, mub, seb = line(db, V.cl)
                sl = ' '.join(f'{nm} {se_cl(db, V.cl, m)[0]:+.5f}({se_cl(db, V.cl, m)[0] / se_cl(db, V.cl, m)[1]:+.1f})'
                              for nm, m in [('M<50', V.M < 50), ('M>=50', V.M >= 50), ('Tue', V.tue), ('Thu', ~V.tue)])
                # matched controls (per-seed mean)
                ctl = []
                for c in ('B0', 'B1'):
                    if c == arm: continue
                    fc = [CACHE / f'oof_{kind}_{c}_{form}_s{s}{sfx}.npy' for s in (1, 2, 3)]
                    if all(f.exists() for f in fc):
                        dc = np.mean(per, 0) - np.mean([np.load(f) for f in fc], 0); a, b = se_cl(dc, V.cl)
                        ctl.append(f'vs {c} {a:+.5f} +- {b:.5f} (z {a / b:+.2f})')
                rho, top = wf_agree(V.off, V.M, bv, g3v if form == 'full' else pto_v)
                out = f'{kind:4s} {tag:8s}{sfx:5s} VAL(seed-mean) {s1} seeds [{seeds}] | bagged {sb} | pmix {pm:+.5f} +- {pse:.5f} | {" ; ".join(ctl)} | {sl} | val rho {rho.mean():.4f} top1 {top.mean():.3f}'
                r = dict(val=mu, val_se=se, val_bag=mub, val_bag_se=seb, pmix=pm, pmix_se=pse, rho_val=float(rho.mean()), top1_val=float(top.mean()))
                bt = bag(kind, tag, 'test', sfx)
                if bt is not None:
                    rt, tt = wf_agree(TE.off, TE.M, bt, g3t if form == 'full' else TE.z(np.asarray(TE.Z('to')).mean(0).astype(np.float32)).astype(np.float64))
                    out += f' | test rho {rt.mean():.4f} top1 {tt.mean():.3f}'; r.update(rho_test=float(rt.mean()), top1_test=float(tt.mean()))
                if form == 'to':
                    fh = [CACHE / f'hold_{kind}_{tag}_s{s}{sfx}.npy' for s in (1, 2, 3)]
                    if all(f.exists() for f in fh):
                        dh = np.mean([np.load(f) for f in fh], 0) - REFH; sh, hm, hse = line(dh, HO.cl)
                        bh = bag(kind, tag, 'hold', sfx); dbh = HO.ev(bh) - REFH; sbh, hbm, hbse = line(dbh, HO.cl)
                        hc = []
                        for c in ('B0', 'B1'):
                            if c == arm: continue
                            fc = [CACHE / f'hold_{kind}_{c}_{form}_s{s}{sfx}.npy' for s in (1, 2, 3)]
                            if all(f.exists() for f in fc):
                                dc = np.mean([np.load(f) for f in fh], 0) - np.mean([np.load(f) for f in fc], 0); a, b = se_cl(dc, HO.cl)
                                hc.append(f'vs {c} {a:+.5f} (z {a / b:+.2f})')
                        out += f'\n        HOLD(seed-mean) {sh} | bagged {sbh} | {" ; ".join(hc)}'
                        r.update(hold=hm, hold_se=hse, hold_bag=hbm, hold_bag_se=hbse)
                pr(out); RES[f'{kind}|{tag}{sfx}'] = r
json.dump(RES, open(OUT / 'e45_eval.json', 'w'), indent=1)
