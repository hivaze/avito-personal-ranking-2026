"""SYNTHESIS check DOMTOK: the ONE labelled read of domtok_predict.py outputs, by the rule in PREDECLARED.txt (written 2026-11-15T22:07:56Z, before any prediction)."""
import sys; sys.path.insert(0, '<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
import pmix
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'; H = R / 'reports/TRICKS_1115/synth_checks/domtok'
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy')
zv = lambda x: feed_rank_z(ov, Mv, np.ascontiguousarray(np.asarray(x, np.float32)))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv).astype(np.float64)
WT = pmix.weights(); WPUB, WPRIV = WT['public'], WT['private']
def f(d, w=None, m=None):
    if m is not None: d = d[m]; w = None if w is None else w[m]
    if w is None: mu, s = d.mean(), d.std() / np.sqrt(len(d))
    else: mu, s = pmix.wmean_se(d, w)
    return f'{mu:+.5f} +- {s:.5f} (z {mu / s:+.2f})'
out = []
def w_(s=''):
    print(s, flush=True); out.append(s)
WIDE = [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in range(951, 959)]
POP = [f'abl_fd10_pop1kd05w192_s{s}' for s in range(951, 955)]
have = [v for v in WIDE if all((H / 'preds' / f'cpu_{v}_{a}.npy').exists() for a in ('base', 'prod', 'mix'))]
w_(f'members with the reference and both settings on disk: {len(have)} of 16')
assert len(have) == 16, 'the rule is read on all 16 members only'
zpop = zv(np.mean([zv(np.load(P / f'nn_{v}_trainonly_val.npy')) for v in POP], 0))
zft = zv(np.load(P / 'ens_FD16P20kd_ft_val.npy'))
def to_branch(kind):
    mem = [np.load(P / f'nn_{v}_trainonly_val.npy') if kind == 'stored' else np.load(H / 'preds' / f'cpu_{v}_{kind}.npy') for v in WIDE]
    return 0.8 * zv(np.mean([zv(x) for x in mem], 0)) + 0.2 * zpop
T = {k: to_branch(k) for k in ('stored', 'base', 'prod', 'mix')}   # 'base' = CPU re-prediction with the stored token = the REFERENCE (amendment in PREDECLARED.txt)
st = np.load(P / 'ens_FD16P20kd_to_val.npy')
w_(f'path check: TO branch rebuilt from stored member files vs preds/ens_FD16P20kd_to_val.npy: corr {np.corrcoef(T["stored"], st)[0, 1]:.7f} max|d| {np.abs(T["stored"] - st).max():.2e}')
for v in WIDE:
    b = H / 'preds' / f'cpu_{v}_base.npy'
    if b.exists():
        r = np.load(P / f'nn_{v}_trainonly_val.npy'); p = np.load(b)
        d = ev(zv(p)) - ev(zv(r)); w_(f'path check: CPU re-prediction with the stored token, {v}: corr {np.corrcoef(r, p)[0, 1]:.7f} | member val NDCG CPU minus stored {d.mean():+.6f}')
F = {k: 0.5 * zft + 0.5 * zv(T[k]) for k in T}
EF = {k: ev(F[k]) for k in F}; ET = {k: ev(T[k]) for k in T}
w_(f'FILE from stored member files: final val {EF["stored"].mean():.5f} (record 0.40271) | TO branch {ET["stored"].mean():.5f}')
w_(f'REFERENCE = FILE from the CPU re-prediction with the stored token: final val {EF["base"].mean():.5f} | TO branch {ET["base"].mean():.5f} | path difference (CPU reference minus stored) FILE {f(EF["base"] - EF["stored"])} | TO {f(ET["base"] - ET["stored"])}')
# cold flags (information): row cold = it_hist_all_luo <= 0
cnt = np.load(R / 'data/feats/val/it_hist_all_luo.npy'); fid = np.repeat(np.arange(len(Mv)), Mv)
npos = np.bincount(fid, yv, len(Mv)); cp = np.bincount(fid, yv * (cnt <= 0), len(Mv))
allcold = (npos > 0) & (cp == npos); nocold = (npos > 0) & (cp == 0)
for k in ('prod', 'mix'):
    d = EF[k] - EF['base']; dt = ET[k] - ET['base']
    w_(); w_(f'## setting {k.upper()}: FILE final val {EF[k].mean():.5f}')
    w_(f'FILE delta: val {f(d)} | pmix-private {f(d, WPRIV)} | pmix-public {f(d, WPUB)} | M<50 {f(d, m=Mv < 50)} | M>=50 {f(d, m=Mv >= 50)}')
    w_(f'  information: TO-branch delta val {f(dt)} | pmix-private {f(dt, WPRIV)} | pmix-public {f(dt, WPUB)}')
    w_(f'  information: FILE delta on feeds whose positives are all cold ({int(allcold.sum())}) {f(d, m=allcold)} | no cold positive ({int(nocold.sum())}) {f(d, m=nocold)}')
    w_(f'  paired public SE of the FILE delta {d.std() / np.sqrt(8562):.5f} | per-feed corr of z, FILE vs reference {np.corrcoef(F[k], F["base"])[0, 1]:.5f}')
    mv, sv = d.mean(), d.std() / np.sqrt(len(d)); mp, sp = pmix.wmean_se(d, WPRIV); ds = d[Mv < 50]; ms_, ss = ds.mean(), ds.std() / np.sqrt(len(ds))
    cont = (mv >= 0) and (mp >= sp) and (ms_ >= -ss); clos = (mp <= 0) or (mv <= -sv)
    w_(f'  RULE: val >= 0 {mv >= 0} | pmix-private >= +1 SE {mp >= sp} | M<50 >= -1 SE {ms_ >= -ss} -> {"CONTINUE" if cont else ("close-clause met" if clos else "INCONCLUSIVE")}')
open(H / 'domtok_read.txt', 'w').write('\n'.join(out) + '\n')
