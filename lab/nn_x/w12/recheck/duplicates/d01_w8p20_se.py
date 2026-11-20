"""Skeptic check (duplicates lens): how large is the paired public SE of the pre-declared W8P20 probe
(per branch 0.8 z(W8 pool) + 0.2 z(pop1kd pool); final 0.5 z(FT) + 0.5 z(TO)) against its +-0.0003 decision thresholds?
CPU, read-only. Val per-feed paired deltas vs the W8 final; SE scaled to the 8,562 public feeds.
FT branch of the pop1kd pool uses the fine-tunes that exist at run time (s951-953 at 20:55Z); TO branch uses all 4 members."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
P = Path('<repo>/preds'); A = Path('<repo>/nn_x/abl/preds')
fv, Mv, ov = feed_meta('val'); ft_, Mt, ot = feed_meta('test'); yv = np.load(FLAT / 'val/cand_target.npy')
Z = {'val': lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1], 'test': lambda x: feed_rank_z(ot, Mt, np.asarray(x, np.float32))[:, 1]}
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
se = lambda d: d.std() / np.sqrt(len(d)); f = lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
NPUB = 8562
def first(*c):
    for p in c:
        if Path(p).exists(): return Path(p)
    return None
out = {}
for sp, ftn, ton in (('val', 'n2cpv3s3_val_oof', 'trainonly_val'), ('test', 'n2cpv3s3_test', 'trainonly_test')):
    z = Z[sp]
    w_ft = np.load(P / f'ens_W8kd_ft_{sp}.npy'); w_to = np.load(P / f'ens_W8kd_to_{sp}.npy')
    to_files = [first(A / f'nn_abl_e10_pop1kd05t25_s{s}_{ton}.npy', P / f'nn_abl_e10_pop1kd05t25_s{s}_{ton}.npy') for s in (951, 952, 953, 954)]
    ft_files = [first(P / f'nn_abl_e10_pop1kd05t25_s{s}_{ftn}.npy', A / f'nn_abl_e10_pop1kd05t25_s{s}_{ftn}.npy') for s in (951, 952, 953, 954)]
    to_files = [p for p in to_files if p is not None]; ft_files = [p for p in ft_files if p is not None]
    if sp == 'val': print(f'pop1kd members found: TO {len(to_files)}, FT {len(ft_files)}')
    p_to = np.mean([z(np.load(p)) for p in to_files], 0); p_ft = np.mean([z(np.load(p)) for p in ft_files], 0)
    out[sp] = dict(w_ft=z(w_ft), w_to=z(w_to), p_ft=z(p_ft), p_to=z(p_to))
fin = lambda sp, w, ftb=True: 0.5 * Z[sp]((1 - w) * out[sp]['w_ft'] + (w * out[sp]['p_ft'] if ftb else w * out[sp]['w_ft'])) + 0.5 * Z[sp]((1 - w) * out[sp]['w_to'] + w * out[sp]['p_to'])
e0 = ev(fin('val', 0.0)); t0 = fin('test', 0.0)
print(f'W8 final val {e0.mean():.5f}')
for w in (0.15, 0.20, 0.30):
    for ftb, name in ((True, 'both branches'), (False, 'TO branch only')):
        d = ev(fin('val', w, ftb)) - e0; c = np.corrcoef(fin('test', w, ftb), t0)[0, 1]
        sp_ = d.std() / np.sqrt(NPUB)
        print(f'w {w:.2f} {name:15s}: val delta {f(d)} (M>=50 {f(d[Mv >= 50])}, M<50 {f(d[Mv < 50])}) | per-feed sd {d.std():.4f} | SE of the public delta (n={NPUB}) {sp_:.5f} | 0.0003 = {0.0003 / sp_:.2f} SE | test corr to W8 {c:.4f}')
