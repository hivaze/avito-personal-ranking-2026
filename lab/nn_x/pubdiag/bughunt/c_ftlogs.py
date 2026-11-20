"""(c) fine-tune outputs/logs of r6 members vs nn/ F7 members: args, feature lists, warnings, json, and the
relation of n2cpv3_test to n2cpv3_test_folds (same rule for both code paths?)."""
import ast, json, re, sys
import numpy as np
sys.path.insert(0, '<repo>/src')
from common import feed_meta
from stack_val import feed_rank_z
R = '<repo>'
roots = range(101, 108)
fams = {'F7': [(f'v17_pre2s_s{r}', 'nn') for r in roots], 'F7x': [(f'r6_cd_m02_r{r}', 'r6') for r in roots],
        'F7x70': [(f'r6_cd_m02_p70k_r{r}', 'r6') for r in roots], 'F7b': [(f'r6_cd_c0_r{r}', 'r6') for r in roots]}
def logdir(p): return f'{R}/nn/logs' if p == 'nn' else f'{R}/nn_x/r6/logs'
ref = None
print('== ft log args / feature lists')
for fam, ms in fams.items():
    for v, p in ms:
        L = open(f'{logdir(p)}/{v}_n2cpv3_ft.log').read().splitlines()
        heads = [i for i, l in enumerate(L) if l.startswith('==== ')]
        if len(heads) != 1: print('  NOTE', v, 'log has', len(heads), 'run headers')
        h = heads[-1]
        args = ast.literal_eval(L[h].split('args: ', 1)[1]); args.pop('variant'); args.pop('out')
        feats = [l for l in L[h:] if l.startswith('extra features (')]
        flist = ast.literal_eval(feats[0].split(': ', 1)[1]) if feats else None
        warn = [l for l in L[h:] if 'missing' in l or 'dropped' in l]
        if ref is None: ref = (args, flist); print('  reference args (from', v, '):', args)
        same_a, same_f = args == ref[0], flist == ref[1]
        j = json.load(open(f'{logdir(p)}/{v}_n2cpv3_ft.json'))
        print(f'  {fam:6s} {v:24s} args_same {same_a} feats_same {same_f} nfeat {len(flist or [])} warnings {len(warn)} '
              f'oof_snap {j["oof_ndcg_snapshot"]:.5f} ema_final {j["oof_ndcg_ema_final"]:.5f} saved {j["saved"]} ep0 {j["oof_by_epoch"][0]:.5f}')
        if not same_a: print('     args diff:', {k: (args.get(k), ref[0].get(k)) for k in set(args) | set(ref[0]) if args.get(k) != ref[0].get(k)})
print('== n2cpv3_test vs test_folds')
_, M, off = feed_meta('test')
z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
for fam, ms in fams.items():
    for v, p in ms[:3]:
        t = np.load(f'{R}/preds/nn_{v}_n2cpv3_test.npy').astype(np.float64)
        tf = np.load(f'{R}/preds/nn_{v}_n2cpv3_test_folds.npy').astype(np.float64)
        fm = tf.mean(0)
        d = np.abs(t - fm)
        zt, zf = z(t), z(fm)
        print(f'  {fam:6s} {v:24s} folds {tf.shape} |test - mean(folds)| max {d.max():.4f} mean {d.mean():.5f} '
              f'corr {np.corrcoef(t, fm)[0,1]:.6f} corr(z) {np.corrcoef(zt, zf)[0,1]:.6f} '
              f'fold-pairwise corr min {min(np.corrcoef(tf)[np.triu_indices(5,1)]):.4f}')
