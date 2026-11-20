"""Recompute every famS number the verdict rests on, with own code, from the saved prediction files."""
import sys; sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review')
from rv_common import *
wv, wh = Wit('val'), Wit('hold'); zwv, zwh = w16(wv), w16(wh)
P = ROOT / 'nn_x/w12/cb/famS/preds'; R = ROOT / 'nn_x/w12/cb/infra/out'
ld = lambda d, t: (np.load(d / f'{t}_val.npy'), np.load(d / f'{t}_hold.npy'))
tags = ['s1_rmse_d8_lr010_s0', 's1f_rmse_d8_lr030_s0', 's1f_rmse_d8_lr030_s1', 's1f_rmse_d8_lr030_s2', 's1f_rmse_d8_lr030_s1long', 's3f_pca_d8_lr030_s0',
        's2_rmse_d10_lr030_s0', 's4_qsm_kdmix_lr030_s0', 's4b_qsm_soft_lr030_s0', 's4y_yeti_teacher_lr030_s0']
X = {t: ld(P, t) for t in tags}; X['ts_lr020_s0'] = ld(R, 'ts_lr020_s0'); X['bundle_file'] = ld(P, 'bundle_s1f_3seeds_z')
res = {}
def do(tag, xv, xh):
    o = read(wv, wh, zwv, zwh, xv, xh, tag); res[tag] = o; print(line(o), flush=True); return o
do('ts_lr020_s0', *X['ts_lr020_s0'])
for t in tags: do(t, *X[t])
S3 = ['s1f_rmse_d8_lr030_s0', 's1f_rmse_d8_lr030_s1', 's1f_rmse_d8_lr030_s2']
do('bundle_s1f_3seeds', [X[t][0] for t in S3], [X[t][1] for t in S3])
do('bundle_file_as_saved', *X['bundle_file'])
do('bundle(s1f_s0,hard)', [X[S3[0]][0], X['ts_lr020_s0'][0]], [X[S3[0]][1], X['ts_lr020_s0'][1]])
bz = {'val': np.mean([wv.z(X[t][0]) for t in S3], 0), 'hold': np.mean([wh.z(X[t][1]) for t in S3], 0)}
do('bundle(3seed_bundle,hard)', [bz['val'], X['ts_lr020_s0'][0]], [bz['hold'], X['ts_lr020_s0'][1]])
# saved bundle file == my z-mean?
print('saved bundle file vs own z-mean: max abs diff val', float(np.abs(wv.z(X['bundle_file'][0]) - wv.z(bz['val'])).max()), 'hold', float(np.abs(wh.z(X['bundle_file'][1]) - wh.z(bz['hold'])).max()))
# joint read
print('--- joint (S = 3-seed bundle, H = hard-label tree)')
J = {}
for w, zw, k in ((wv, zwv, 'val'), (wh, zwh, 'hold')):
    zs = bz[k]; zh = w.z(X['ts_lr020_s0'][0 if k == 'val' else 1])
    _, lw = w.clogit(zw); _, ls = w.clogit(np.column_stack([zw, zs])); _, lh = w.clogit(np.column_stack([zw, zh])); b3, lsh = w.clogit(np.column_stack([zw, zs, zh]))
    f = lambda d: (float(d.mean()), se(d))
    J[k] = dict(WS=f(lw - ls), WH=f(lw - lh), WSH=f(lw - lsh), S_given_WH=f(lh - lsh), H_given_WS=f(ls - lsh), beta3=[float(v) for v in b3], corrSH=float(np.corrcoef(zs, zh)[0, 1]))
    print(k, {a: (f'{v[0]:+.5f}+-{v[1]:.5f}' if isinstance(v, tuple) else v) for a, v in J[k].items()}, flush=True)
res['joint'] = J
# paired level reads
print('--- paired level differences')
def cmp(name, a, b):
    dv = wv.ndcg(a[0]) - wv.ndcg(b[0]); dh = wh.ndcg(a[1]) - wh.ndcg(b[1]); z = (dv.mean() / se(dv) + dh.mean() / se(dh)) / np.sqrt(2)
    res['cmp_' + name] = dict(val=float(dv.mean()), val_se=se(dv), hold=float(dh.mean()), hold_se=se(dh), z=float(z)); print(f'{name}: {dv.mean():+.5f}+-{se(dv):.5f} | {dh.mean():+.5f}+-{se(dh):.5f} | z {z:+.2f}', flush=True)
B = (bz['val'], bz['hold'])
cmp('bundle - hard', B, X['ts_lr020_s0']); cmp('S1 - hard', X[tags[0]], X['ts_lr020_s0']); cmp('S1f_s0 - hard', X[tags[1]], X['ts_lr020_s0']); cmp('S4b - hard', X['s4b_qsm_soft_lr030_s0'], X['ts_lr020_s0'])
cmp('S4b - S1f_s0', X['s4b_qsm_soft_lr030_s0'], X[tags[1]]); cmp('S4b - bundle', X['s4b_qsm_soft_lr030_s0'], B); cmp('S3 - S1f_s0', X['s3f_pca_d8_lr030_s0'], X[tags[1]]); cmp('S2 - S1f_s0', X['s2_rmse_d10_lr030_s0'], X[tags[1]])
cmp('S1long - bundle', X['s1f_rmse_d8_lr030_s1long'], B); cmp('S4b - S1long', X['s4b_qsm_soft_lr030_s0'], X['s1f_rmse_d8_lr030_s1long'])
# gap to W16 by M bin (bundle) and val M>=50 nats / member rule
print('--- M bins (bundle)')
e0 = wv.ndcg(zwv); eb = wv.ndcg(bz['val']); g = e0 - eb
for nm, m in (('val M<50', wv.M < 50), ('val M>=50', wv.M >= 50)):
    print(nm, int(m.sum()), f'gap {g[m].mean():.5f}+-{se(g[m]):.5f} level {eb[m].mean():.5f} W16 {e0[m].mean():.5f}')
gh = wh.ndcg(zwh) - wh.ndcg(bz['hold']); print('holdout gap', f'{gh.mean():.5f}+-{se(gh):.5f}'); print('val gap all', f'{g.mean():.5f}+-{se(g):.5f}')
res['gap'] = dict(val=float(g.mean()), val_se=se(g), hold=float(gh.mean()), hold_se=se(gh), lt50=float(g[wv.M < 50].mean()), ge50=float(g[wv.M >= 50].mean()))
# val M>=50: refit clogit on those feeds only
class Sub(Wit):
    def __init__(self, w, m):
        rows = np.repeat(m, w.M); self.sp = w.sp; self.M = w.M[m]; self.off = np.concatenate([[0], np.cumsum(self.M)[:-1]]); self.y = w.y[rows]; self.nf = len(self.M); self.fid = np.repeat(np.arange(self.nf), self.M)
        self.npos = np.bincount(self.fid, self.y, self.nf); self.rank0 = np.arange(len(self.y)) - np.repeat(self.off, self.M); self.disc = 1.0 / np.log2(self.rank0 + 2.0)
        self.idcg = np.bincount(self.fid, (self.rank0 < np.repeat(self.npos, self.M)) * self.disc, self.nf); self.rows = rows
for nm, m in (('val M>=50', wv.M >= 50), ('val M<50', wv.M < 50)):
    s = Sub(wv, m)
    for tg, zx in (('bundle', bz['val']), ('hard', wv.z(X['ts_lr020_s0'][0])), ('S4b', wv.z(X['s4b_qsm_soft_lr030_s0'][0]))):
        n, sn, _, _ = s.nats(zwv[s.rows], [zx[s.rows]]); dm = s.ndcg(0.9 * zwv[s.rows] + 0.1 * zx[s.rows]) - s.ndcg(zwv[s.rows])
        print(f'{nm} {tg}: NATS {n:+.5f}+-{sn:.5f} member {dm.mean():+.5f}+-{se(dm):.5f}', flush=True); res[f'{nm}|{tg}'] = dict(nats=n, nats_se=sn, mix=float(dm.mean()), mix_se=se(dm))
# other weights for the bundle (reported, never chosen) + the best weight in hindsight (information)
print('--- weights (bundle)')
for wgt in (0.02, 0.05, 0.1, 0.2, 0.3):
    dv = wv.ndcg((1 - wgt) * zwv + wgt * wv.z(bz['val'])) - e0; dh = wh.ndcg((1 - wgt) * zwh + wgt * wh.z(bz['hold'])) - wh.ndcg(zwh)
    print(f'w {wgt}: {dv.mean():+.5f}+-{se(dv):.5f} | {dh.mean():+.5f}+-{se(dh):.5f}')
json.dump(res, open(HERE / 'out/famS_own.json', 'w'), indent=1)
