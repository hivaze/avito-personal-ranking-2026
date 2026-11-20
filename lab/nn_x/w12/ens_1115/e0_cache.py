"""e0: build the z cache of every 10-epoch-class member on disk + the raw inventory table (out/inventory.json).
Member = any name that has BOTH a train-only val and a train-only test file in preds/, nn_x/w5/preds/ or nn_x/abl/preds/ and matches the class filter below.
For every member and kind: path used (preds/ first), duplicate copies checked for equality, per-feed z written to cache/z_<kind>.f32.
Run: cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... .venv/bin/python nn_x/w12/ens_1115/e0_cache.py"""
from ens_common import *
import time, hashlib
from concurrent.futures import ThreadPoolExecutor
t0 = time.time(); m = Meta()
INC = re.compile(r'^(w5_e10_|w5_fd10_|w5_fdx|w5_[kl]cd_|abl_e10_|abl_fd10_|r6_cd_m02|v23_gru_pre2s|v17_pre2s|v22_pre2s_lr1|v26_d192_pre2s|v14_)')
names = set()
for d in (P, W, AB):
    for f in os.listdir(d):
        if f.startswith('nn_') and f.endswith('_trainonly_val.npy'): names.add(f[3:-len('_trainonly_val.npy')])
names = sorted(n for n in names if INC.match(n) and 'smoke' not in n and find(n, 'to_test'))
# @last variants: kept only for the re-cool arms (decision checkpoints) and for members that have a fine-tune of the @last checkpoint
names = [n for n in names if ('@last' not in n) or re.match(r'^w5_[kl]cd_', n) or find(n, 'ft_val')]
print(len(names), 'members', flush=True)
idx = {k: {} for k in KINDS}; files = {}
for k in KINDS:
    for n in names:
        fs = find(n, k)
        if k in ('ft_val', 'ft_test') and not (find(n, 'ft_val') and find(n, 'ft_test')): continue
        if fs: idx[k][n] = len(idx[k]); files[(k, n)] = fs
print({k: len(v) for k, v in idx.items()}, flush=True)
mm = {k: np.memmap(CACHE / f'z_{k}.f32', np.float32, 'w+', shape=(len(idx[k]), m.n[SPLIT[k]])) for k in KINDS}
dup_bad = []; info = {}

def work(key):
    k, n = key; fs = files[key]; x = np.load(fs[0]).astype(np.float32)
    assert x.shape == (m.n[SPLIT[k]],) and np.isfinite(x).all(), (key, x.shape)
    for f2 in fs[1:]:
        y = np.load(f2).astype(np.float32)
        if y.shape != x.shape or not np.array_equal(x, y): dup_bad.append((str(fs[0]), str(f2), float(np.corrcoef(x, y)[0, 1]) if y.shape == x.shape else None))
    z = m.z(SPLIT[k], x); mm[k][idx[k][n]] = z
    nd = float(m.ev(z).mean()) if SPLIT[k] == 'val' else (float(m.eh(z).mean()) if k == 'hold' else None)
    return key, dict(path=str(fs[0].relative_to(ROOT)), copies=len(fs), mtime=os.path.getmtime(fs[0]), ndcg=nd)

keys = list(files)
with ThreadPoolExecutor(8) as ex:
    for i, (key, d) in enumerate(ex.map(work, keys)):
        info[key] = d
        if i % 200 == 0: print(i, len(keys), f'{time.time() - t0:.0f}s', flush=True)
for k in KINDS: mm[k].flush()
json.dump(idx, open(CACHE / 'index.json', 'w'))
# full-data flag from the training log json (args.holdout_in_train), as the freeze audit reads it
def full(n):
    b = n.replace('@last', '')
    for d in ('nn_x/w5/logs', 'nn_x/abl/logs', 'nn/logs', 'nn_x/r6/logs'):
        f = ROOT / d / f'{b}.json'
        if f.exists():
            try: return bool(json.load(open(f)).get('args', {}).get('holdout_in_train', False)), str(f.relative_to(ROOT))
            except Exception: pass
    return None, None
inv = {}
for n in names:
    fl, lg = full(n)
    inv[n] = dict(arm=arm_of(n), full_data=fl, log=lg, **{k: info.get((k, n)) for k in KINDS})
json.dump(dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), members=inv, dup_bad=dup_bad), open(OUT / 'inventory.json', 'w'), indent=0)
print('duplicate copies that differ:', len(dup_bad)); [print('  ', d) for d in dup_bad[:40]]
print(f'done {time.time() - t0:.0f}s')
