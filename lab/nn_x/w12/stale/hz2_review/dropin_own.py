"""hz2_review R5: the reviewer's own identity test of the drop-in (CPU). Own split object (not the study's MiniSplit), real files, FULL train row space.
 A = nn_x/w5/cd_arms6.py, B = nn_x/w5/cd_arms6b.py (the copy the live lanes import) with HZCD_DIR / HZ2_KEEP unset, C = the same file with the hz2 lane environment.
 Checks: (1) A and B: same column list, bit-identical hz tensor over all 246.7M rows, bit-identical gathered batches on 200 random train feeds.
         (2) C: every swapped cell over ALL rows = fp16(attach_xt transform of the view file), except keep rows = stored xt; untouched: nothing else exists in this split.
Usage: CUDA_VISIBLE_DEVICES="" python dropin_own.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
import sys, time, importlib.util, types
sys.dont_write_bytecode = True
import numpy as np
import polars as pl
import torch
torch.set_num_threads(12)

R = '<repo>'
T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


NAMES = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]   # preset cvraw = xt columns 0-7 of cvnd_pre2s
ftr = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
M = ftr['M'].to_numpy().astype(np.int64); off = ftr['cand_off'].to_numpy().astype(np.int64)
NC = int(M.sum())


def tf_stored(x):                       # data.SplitData.attach_xt branch of seq_cv* / it_hist*: log1p(max(x, 0)), NaN -> 0, fp16
    return torch.from_numpy(np.nan_to_num(np.log1p(np.maximum(x.astype(np.float32), 0)), nan=0.0)).to(torch.float16)


class Split:
    """Minimal stand-in for nn_x/w5/data.SplitData: xt [n_cand, 8] fp16 and the gather() contract HzTrainView relies on (b['ci'], b['xt'] = xt[ci].float())."""
    def __init__(self):
        self.device = torch.device('cpu'); self.n_cand = NC
        self.xt = torch.stack([tf_stored(np.load(f'{R}/nn/feat_cache/train/{n}.npy')) for n in NAMES], 1)
        self.M_np = M; self.cand_off = torch.from_numpy(off); self.Mt = torch.from_numpy(M)

    def gather(self, fidx_np, with_pos=False):
        fidx = torch.from_numpy(np.asarray(fidx_np, dtype=np.int64))
        Mb = self.Mt[fidx]; Mx = int(-(-self.M_np[fidx_np].max() // 16) * 16)
        ar = torch.arange(Mx)
        cmask = ar[None, :] < Mb[:, None]
        ci = self.cand_off[fidx][:, None] + ar[None, :]
        ci = torch.where(cmask, ci, self.cand_off[fidx][:, None])
        return dict(fidx=fidx, M=Mb, cmask=cmask, ci=ci, xt=self.xt[ci].float())


def load_mod(path, tag, env):
    for k in ('HZCD_DIR', 'HZ2_KEEP'):
        os.environ.pop(k, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


tr = Split(); lg('stored xt built', tuple(tr.xt.shape), tr.xt.dtype)
a = types.SimpleNamespace(hcap='', hzcd=True)
LOG = {}
A = load_mod(f'{R}/nn_x/w5/cd_arms6.py', 'cdA', {})
vA = A.setup_hzcd(a, tr, NAMES, np.arange(5), 0, 10, lambda s: LOG.__setitem__('A', s)); lg('A view', [c[1] for c in vA.cols])
B = load_mod(f'{R}/nn_x/w5/cd_arms6b.py', 'cdB', {})
vB = B.setup_hzcd(a, tr, NAMES, np.arange(5), 0, 10, lambda s: LOG.__setitem__('B', s)); lg('B view', [c[1] for c in vB.cols], '| HZ_DIR', B.HZ_DIR, '| KEEP_PATH', repr(B.KEEP_PATH))
ok_cols = [tuple(c) for c in vA.cols] == [tuple(c) for c in vB.cols]
ok_hz = bool(torch.equal(vA.hz, vB.hz)) and vA.hz.dtype == vB.hz.dtype
lg(f'(1) same columns: {ok_cols}; hz tensors bit-identical over {tuple(vA.hz.shape)}: {ok_hz}')
rng = np.random.default_rng(777)
feeds200 = rng.choice(len(M), 200, replace=False)
okb = True; nb = 0; nrows = 0
for i in range(0, 200, 25):
    fb = feeds200[i:i + 25]
    bA = vA.gather(fb); bB = vB.gather(fb)
    for k in bA:
        okb &= bool(torch.equal(bA[k], bB[k])) and bA[k].dtype == bB[k].dtype
    okb &= (bA['xt'].numpy().tobytes() == bB['xt'].numpy().tobytes())
    nb += 1; nrows += int(M[fb].sum())
lg(f'(1) gathered batches bit-identical (cd_arms6 vs cd_arms6b, variables unset): {okb} on {nb} batches, 200 random train feeds, {nrows} rows')
# the plain gather must differ from the view gather in the 5 LUO columns only
b0 = tr.gather(feeds200[:25]); b1 = vB.gather(feeds200[:25])
lg('    columns changed by the hz view on the first batch:', [NAMES[c] for c in range(8) if not torch.equal(b0['xt'][..., c], b1['xt'][..., c])])
_b = LOG['B']; _i = _b.find('[6b:'); _j = _b.find('] ', _i); _bw = _b[:_i] + _b[_j + 2:]
lg('    log lines differ only by the [6b: ...] tag:', LOG['A'].split(' GB; ')[0] == _bw.split(' GB; ')[0], '| tag:', _b[_i:_j + 1][:200])
del vA
V = f'{R}/nn_x/w12/stale/hz2/feats'
C = load_mod(f'{R}/nn_x/w5/cd_arms6b.py', 'cdC', {'HZCD_DIR': V, 'HZ2_KEEP': f'{V}/train/hz2_keep_rows.npy'})
vC = C.setup_hzcd(a, tr, NAMES, np.arange(5), 0, 10, lambda s: LOG.__setitem__('C', s))
lg('C view columns', [(c[0], c[1], c[2]) for c in vC.cols])
lg('C log:', LOG['C'][LOG['C'].find('[6b:'):][:900])
keep = torch.from_numpy(np.load(f'{V}/train/hz2_keep_rows.npy'))
lg('keep rows', int(keep.sum()))
allok = True
for k, (c, sn, hn) in enumerate(vC.cols):
    exp = tf_stored(np.load(f'{V}/train/{hn}.npy'))
    exp = torch.where(keep, tr.xt[:, c], exp)
    okc = bool(torch.equal(vC.hz[:, k], exp)); allok &= okc
    ch = float((vC.hz[:, k] != tr.xt[:, c]).float().mean()); z0 = float((tr.xt[:, c] == 0).float().mean()); z1 = float((vC.hz[:, k] == 0).float().mean())
    fin = bool(torch.isfinite(vC.hz[:, k]).all())
    lg(f'(2) col {c} {sn} <- {hn}: every row = keep ? stored : fp16(log1p(view file)): {okc}; finite: {fin}; rows changed vs stored {ch:.4f}; zero rate {z0:.3f} -> {z1:.3f}')
lg(f'(2) all 8 columns as specified over all {NC} rows: {allok}')
# a holdout-style gather through C on the 200 feeds: non-swapped keys identical to the plain gather
bC = vC.gather(feeds200[:25]); b0 = tr.gather(feeds200[:25])
lg('    keys other than xt identical to the plain gather:', all(bool(torch.equal(bC[k], b0[k])) for k in bC if k != 'xt'))
lg('DROPIN_OWN DONE')
