"""PMIX: the val reads of the surviving blocks reweighted to the private (and public) feed mix (src/pmix.weights: M bin x cold-share bin).
Per-feed NATS gains from the unweighted fit (composite coefficients from all holdout feeds), then the weighted mean; also the tree-setting
contrast B3 - B0 (per-feed NATS over W16, seeds 0 and 1 averaged).  -> out/pmix_read.txt"""
from cbi_common import *
import json
sys.path.insert(0, '<repo>/src'); import pmix
meta = Meta(); WV = Wit(meta, 'val'); WH = Wit(meta, 'hold'); PW = pmix.weights()
FB = ROOT / 'nn_x/w12/cb/famB'; FBN = json.load(open(FB / 'feat/c_names.json'))
def feat(name, w):
    x = np.load(FEATS / ('val' if w.sp == 'val' else 'train') / f'{name}.npy', mmap_mode='r'); return np.asarray(x if w.rows is None else x[w.rows], np.float32)
def kn(name, w):
    a = np.load(FB / f"feat/c_{'val' if w.sp == 'val' else 'hold'}.npy", mmap_mode='r'); return np.asarray(a[:, FBN.index(name)], np.float32)
def wm(d, w):
    m, s = pmix.wmean_se(d, w); return f'{m:+.5f} +- {s:.5f}'
lines = [f"cold share: val {PW['cs_val']:.3f}, public {PW['cs_pub']:.3f}, private {PW['cs_priv']:.3f}"]
SETS = {'ctr (xp_ctr_c)': [('xp_ctr_c', feat)], 'density (kn_cos1, kn_cos20)': [('kn_cos1', kn), ('kn_cos20', kn)],
        'ctr + density': [('xp_ctr_c', feat), ('kn_cos1', kn), ('kn_cos20', kn)], 'tte_it_coec': [('tte_it_coec', feat)]}
for k, cols in SETS.items():
    Xh = np.column_stack([WH.zcol(ld(c, WH)) for c, ld in cols]); bh, _ = clogit(np.column_stack([WH.zw16, Xh]), WH, None, ridge=1e-6)
    comp = np.column_stack([WV.zcol(ld(c, WV)) for c, ld in cols]) @ bh[1:]; d, b2 = nats(comp, WV.zw16, WV)
    mix = WV.ndcg(b2[0] * WV.zw16 + b2[1] * comp) - WV.ndcg(WV.zw16)
    lines.append(f"{k:28s} | val NATS plain {f2(d)} | private mix {wm(d, PW['private'])} | public mix {wm(d, PW['public'])} | linear dNDCG plain {f2(mix)} | private mix {wm(mix, PW['private'])}")
    print(lines[-1], flush=True)
z = lambda p: WV.z(np.load(ROOT / p))
b0 = [nats(z(p), WV.zw16, WV)[0] for p in ('nn_x/w12/cb/infra/out/ts_lr020_s0_val.npy', 'nn_x/w12/cb/famB/preds/b0_s1_val.npy')]
b3 = [nats(z(p), WV.zw16, WV)[0] for p in ('nn_x/w12/cb/famB/preds/b3_abc_s0_val.npy', 'nn_x/w12/cb/famB/preds/b3_abc_s1_val.npy')]
dd = np.mean(b3, 0) - np.mean(b0, 0)
lines.append(f"tree setting, single-model NATS of B3 minus B0 (mean of seeds 0, 1; per feed) | val plain {f2(dd)} | private mix {wm(dd, PW['private'])} | public mix {wm(dd, PW['public'])} | M >= 50 {f2(dd[WV.M >= 50])} | M < 50 {f2(dd[WV.M < 50])}")
print(lines[-1]); open(CBI_OUT + '/pmix_read.txt', 'w').write('\n'.join(lines) + '\n')
