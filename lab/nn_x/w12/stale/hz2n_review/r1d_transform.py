"""R1d: cd_arms6c.xt_transform(hz name, x) vs the branch data.SplitData.attach_xt applies to the STORED name, on real values (first 3M rows of win1 + 1M rows before the windows: NaN rows included)."""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
R = '<repo>'
sys.path.insert(0, f'{R}/nn_x/w5')
import cd_arms6c as m
N = f'{R}/nn_x/w12/stale/hz2n'
def attach_branch(n, x):  # nn_x/w5/data.py attach_xt lines 191-198, by the STORED name (has: handled by the caller)
    x = x.astype(np.float32)
    if n.startswith("has:") or n.endswith("_rk"): pass
    elif n.endswith("_z"): x = np.clip(x, -5, 5)
    elif n.startswith(("seq_cv", "it_hist")): x = np.log1p(np.maximum(x, 0))
    else: x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)
MAP = {**m.LUO_TO_HZ, **m.EXTRA_TO_HZ, **m.ENC_TO_HZ}
src = {'pre2_hz_fe_has': f'{N}/feats_n/aux', 'pre2_hz_new': f'{N}/feats_nF/train', 'pre2_hz_new_z': f'{N}/feats_nF/train', 'pre2_hz_new_s': f'{N}/feats_nF/train'}
rows = np.r_[np.arange(0, 1_000_000), np.arange(80_000_000, 83_000_000)]
out = {}
for st, hz in MAP.items():
    d = src.get(hz, f'{N}/feats_n/train')
    p = f'{d}/{hz}.npy'
    if not os.path.exists(p):
        out[st] = 'no file'; continue
    x = np.load(p, mmap_mode='r')[rows]
    a = m.xt_transform(hz, x); b = attach_branch(st, np.asarray(x))
    out[st] = dict(hz=hz, rows=len(rows), nan_in=int(np.isnan(np.asarray(x, np.float32)).sum()), bits_diff=int((a.view(np.int32) != b.view(np.int32)).sum()), min=float(np.nanmin(x)), max=float(np.nanmax(x)))
out['R1d_pass'] = all(v == 'no file' or v['bits_diff'] == 0 for v in out.values())
json.dump(out, open(f'{R}/nn_x/w12/stale/hz2n_review/r1d_transform.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
