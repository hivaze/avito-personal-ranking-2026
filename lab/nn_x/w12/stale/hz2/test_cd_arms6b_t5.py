"""T5 (resumed run): the BODY of cd_arms6b.setup_hzcd executed end to end on a CPU MiniSplit of 200 random train feeds, with HZCD_DIR = the hz2 view and HZ2_KEEP set
(the keep branch of setup_hzcd was only desk-checked in T4, which called apply_keep directly). load_hz is wrapped so that it reads the sample rows (rows=crow); the keep file is the
sample's slice of hz2_keep_rows.npy written to cache/t5_keep_sample.npy. Checks: 8 swapped columns; the log line carries the [6b: ...] tag with the right keep count; every batch
through the returned view equals stored values on keep rows and fp16(transform(hz2 file)) on the other rows; every other key / column is the plain gather.
Also T5b: setup_hzcd with BOTH variables unset returns a view whose batches are bit-identical to cd_arms6.setup_hzcd's (the exact call train.py makes).
Usage: CUDA_VISIBLE_DEVICES="" python test_cd_arms6b_t5.py -> logs/test_cd_arms6b_t5.log"""
import os, sys, types
sys.dont_write_bytecode = True
import numpy as np
import torch
torch.set_num_threads(12)
import test_cd_arms6b as T
import polars as pl
from mini_split import MiniSplit
HERE, ROOT, W5 = T.HERE, T.ROOT, T.W5
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); out.append(s); print(s, flush=True)
f = pl.read_parquet(f'{ROOT}/data/flat/train/feeds.parquet')
feeds = np.sort(np.random.default_rng(777).choice(len(f), 200, replace=False))     # a NEW sample (T2-T4 used seed 20261115)
M = f['M'].to_numpy().astype(np.int64)[feeds]; coff = f['cand_off'].to_numpy().astype(np.int64)[feeds]
crow = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])
xt_names = T.xt_presets()['cvnd_pre2s']
ms = MiniSplit('train', feeds, xt_names=xt_names)
VAR = f'{HERE}/feats'
keep_np = np.asarray(np.load(f'{VAR}/train/hz2_keep_rows.npy', mmap_mode='r')[crow])
kp = f'{HERE}/cache/t5_keep_sample.npy'; np.save(kp, keep_np)
a = types.SimpleNamespace(hcap='', variant='t5')
logs = []
fb = [np.arange(s, min(200, s + 25)) for s in range(0, 200, 25)]
# ---- T5: hz2 view + keep through setup_hzcd
B = T.load_mod(f'{HERE}/cd_arms6b.py', 'cd_arms6b_t5', {'HZCD_DIR': VAR, 'HZ2_KEEP': kp})
assert B.HZ_DIR == VAR and B.KEEP_PATH == kp
orig = B.load_hz
B.load_hz = lambda split, cols, device, rows=None: orig(split, cols, device, rows=crow)
view = B.setup_hzcd(a, ms, xt_names, np.arange(5), 10, 100, logs.append)
assert [c[0] for c in view.cols] == list(range(8))
line = logs[-1]
assert f'[6b: 8 swapped columns' in line and f'HZ2_KEEP rows on stored values: {int(keep_np.sum())}]' in line and VAR in line, line
brute = torch.stack([torch.from_numpy(B.xt_transform(hn, np.asarray(np.load(f'{VAR}/train/{hn}.npy', mmap_mode='r')[crow]))).to(torch.float16) for _, _, hn in view.cols], 1)
keep = torch.from_numpy(keep_np)
want_all = torch.where(keep[:, None], ms.xt[:, :8], brute)
n_keep_cells = n_hz_cells = 0
for ids in fb:
    b, b0 = view.gather(ids, with_pos=True), ms.gather(ids, with_pos=True)
    assert not T.batches_equal(b, b0, skip_xt_cols=list(range(8)))
    assert T.bits_equal(b['xt'][..., :8].contiguous(), want_all[b['ci']].float().contiguous())
    kk = keep[b['ci']]
    n_keep_cells += int(kk.sum()) * 8; n_hz_cells += int((~kk).sum()) * 8
    assert T.bits_equal(b['xt'][..., :8][kk].contiguous(), b0['xt'][..., :8][kk].contiguous())
P(f'T5 PASS: setup_hzcd with HZCD_DIR = hz2 view and HZ2_KEEP on {len(feeds)} NEW random train feeds ({len(crow)} rows, {int(keep_np.sum())} keep rows): 8 columns swapped; '
  f'{len(fb)} batches: keep rows == stored xt, other rows == fp16(attach_xt transform(hz2 file)); every other key / column == plain gather; log tag present')
P('   log line: ' + line[:420] + ' ...')
# ---- T5b: both variables unset, setup_hzcd of both modules
C6 = T.load_mod(f'{W5}/cd_arms6.py', 'cd_arms6_t5', {})
B0 = T.load_mod(f'{HERE}/cd_arms6b.py', 'cd_arms6b_t5b', {})
o6, ob = C6.load_hz, B0.load_hz
C6.load_hz = lambda split, cols, device, rows=None: o6(split, cols, device, rows=crow)
B0.load_hz = lambda split, cols, device, rows=None: ob(split, cols, device, rows=crow)
l6, lb = [], []
v6 = C6.setup_hzcd(a, ms, xt_names, np.arange(5), 10, 100, l6.append); vb = B0.setup_hzcd(a, ms, xt_names, np.arange(5), 10, 100, lb.append)
assert v6.cols == vb.cols and T.bits_equal(v6.hz, vb.hz)
for ids in fb:
    assert not T.batches_equal(v6.gather(ids, with_pos=True), vb.gather(ids, with_pos=True))
assert '[6b: 5 swapped columns' in lb[-1] and 'HZ2_KEEP rows on stored values: 0]' in lb[-1]
P(f'T5b PASS: both variables unset: cd_arms6.setup_hzcd and cd_arms6b.setup_hzcd return the same columns {[c[1] for c in vb.cols]}, bit-identical hz tensors and {len(fb)} bit-identical batches on the new sample; '
  f'the 6b log line differs from the cd_arms6 line only by the inserted [6b: ...] tag: {lb[-1].replace("[6b: 5 swapped columns " + str([c[1] for c in vb.cols]) + "; HZ2_KEEP rows on stored values: 0] ", "").split("; hz tensor")[0] == l6[-1].split("; hz tensor")[0]}')
open(f'{HERE}/logs/test_cd_arms6b_t5.log', 'w').write('\n'.join(out) + '\n')
