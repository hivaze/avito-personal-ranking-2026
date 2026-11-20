"""ft_1115 step A: exact inventory of the stored fine-tune outputs of the members of W8P20 and FD16P20, and the per-epoch
out-of-fold NDCG curves from the fine-tune logs (json: oof_by_epoch = raw weights, mean of the 3 seeds; fold_curves = seed 0).
CPU only, read-only. Output: out/a_inventory.txt, out/a_curves.json"""
import json, sys, os
import numpy as np
from pathlib import Path
R = Path('<repo>'); P = R / 'preds'
S8 = range(951, 959); S4 = range(951, 955)
POOLS = {
    'W8P20 base (8 wide students)': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4],
    'W8P20 pop-light (4 d128 students)': [f'abl_e10_pop1kd05t25_s{s}' for s in S4],
    'FD16P20 base (16 full-data wide students)': [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'FD16P20 pop-light (4 full-data wide students)': [f'abl_fd10_pop1kd05w192_s{s}' for s in S4],
}
HARN = {'w5': (R / 'nn_x/w5/preds', R / 'nn_x/w5/logs'), 'abl': (R / 'nn_x/abl/preds', R / 'nn_x/abl/logs')}
KINDS = ['n2cpv3s3_val_oof', 'n2cpv3s3_test', 'n2cpv3s3_test_folds', 'trainonly_val', 'trainonly_test']
out = []; curves = {}
def w(s=''):
    print(s); out.append(s)
for pool, names in POOLS.items():
    w(f'## {pool}')
    w('member | preds/: ' + ' '.join(KINDS) + ' | harness preds: same | ft log/json | per-seed files | per-epoch files')
    for m in names:
        h = 'w5' if m.startswith('w5_') else 'abl'; hp, hl = HARN[h]
        a = ''.join('Y' if (P / f'nn_{m}_{k}.npy').exists() else '-' for k in KINDS)
        b = ''.join('Y' if (hp / f'nn_{m}_{k}.npy').exists() else '-' for k in KINDS)
        lg = hl / f'{m}_n2cpv3s3_ft.log'; js = hl / f'{m}_n2cpv3s3_ft.json'
        # any other file that mentions the member and a fine-tune (per-seed / per-epoch / 1-seed variants)
        other = sorted({f.name for d in (P, hp) for f in d.glob(f'nn_{m}_*') if not any(f.name == f'nn_{m}_{k}.npy' for k in KINDS)})
        w(f'{m} | {a} | {b} | log {"Y" if lg.exists() else "-"} json {"Y" if js.exists() else "-"} | other files: {other if other else "none"}')
        if js.exists():
            j = json.load(open(js)); curves[m] = dict(oof_by_epoch=j['oof_by_epoch'], fold_curves=j['fold_curves'], ema=j['oof_ndcg_ema_final'],
                                                      snap=j['oof_ndcg_snapshot'], best=j['best_epoch'], args=j['args'], minutes=j['minutes'])
    w()
# shapes of one member's files
m = POOLS['FD16P20 base (16 full-data wide students)'][0]
for k in KINDS:
    f = (P / f'nn_{m}_{k}.npy'); f = f if f.exists() else HARN['w5'][0] / f'nn_{m}_{k}.npy'
    x = np.load(f, mmap_mode='r'); w(f'shape {k}: {x.shape} {x.dtype} ({f})')
json.dump(curves, open(R / 'nn_x/w12/ft_1115/out/a_curves.json', 'w'))
# ---- curves
def block(title, names):
    names = [n for n in names if n in curves]
    O = np.array([curves[n]['oof_by_epoch'] for n in names])            # members x 5 (epoch 0 = train-only)
    ema = np.array([curves[n]['ema'] for n in names]); snap = np.array([curves[n]['snap'] for n in names])
    D = O - O[:, :1]
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
    w(f'### {title}: {len(names)} members; out-of-fold NDCG minus the train-only model (epoch 0), mean over members +- SE across members')
    w('  epoch 1..4 (raw weights, mean of 3 seeds): ' + ' | '.join(f'e{e} {D[:, e].mean():+.5f}+-{se(D[:, e]):.5f}' for e in range(1, 5)))
    w(f'  saved (final EMA weights): {(ema - O[:, 0]).mean():+.5f}+-{se(ema - O[:, 0]):.5f} | mean of epochs 1..4 (per-feed z): {(snap - O[:, 0]).mean():+.5f}+-{se(snap - O[:, 0]):.5f}')
    w(f'  epoch 1 minus saved: {(O[:, 1] - ema).mean():+.5f}+-{se(O[:, 1] - ema):.5f} ({int((O[:, 1] > ema).sum())}/{len(names)} members > 0) | epoch-mean minus saved: {(snap - ema).mean():+.5f}+-{se(snap - ema):.5f} ({int((snap > ema).sum())}/{len(names)})')
    w(f'  best epoch (member-level OOF argmax over 0..4): ' + ' '.join(f'e{e}:{int((O.argmax(1) == e).sum())}' for e in range(5)))
    F = np.array([curves[n]['fold_curves'] for n in names])               # members x folds x 5 (seed 0)
    am = F.argmax(2)
    w(f'  fold curves (seed 0), {F.shape[0] * F.shape[1]} member-folds: argmax epoch ' + ' '.join(f'e{e}:{int((am == e).sum())}' for e in range(5)))
    FD = F - F[:, :, :1]
    w('  fold-level delta vs epoch 0, mean over members, by fold: ' + ' ; '.join(f'fold {k}: ' + ' '.join(f'{FD[:, k, e].mean():+.4f}' for e in range(1, 5)) for k in range(F.shape[1])))
    w(f'  fine-tune wall time per member (3 seeds x 5 folds): mean {np.mean([curves[n]["minutes"] for n in names]):.1f} min (shared GPU)')
    return names
w('## Per-epoch out-of-fold curves from the fine-tune logs')
for pool, names in POOLS.items():
    block(pool, names)
block('FD16P20 all 20 members', POOLS['FD16P20 base (16 full-data wide students)'] + POOLS['FD16P20 pop-light (4 full-data wide students)'])
block('W8P20 all 12 members', POOLS['W8P20 base (8 wide students)'] + POOLS['W8P20 pop-light (4 d128 students)'])
# args identical?
keys = ['epochs', 'lr', 'wd', 'warmup', 'cand_budget', 'max_feeds', 'loss', 'loss_norm', 'folds', 'seeds', 'extra', 'extra_lr_mult', 'decay', 'ema', 'ema_start', 'u0_singleton', 'select', 'pseudo', 'add_random', 'drop', 'from_scratch']
ref = None; bad = []
for n, c in curves.items():
    a = {k: c['args'].get(k) for k in keys}
    if ref is None: ref = a
    elif a != ref: bad.append(n)
w(f'logged fine-tune args identical over the {len(curves)} members on {len(keys)} keys: {"YES" if not bad else "NO: " + str(bad)} -> {ref}')
open(R / 'nn_x/w12/ft_1115/out/a_inventory.txt', 'w').write('\n'.join(out) + '\n')
