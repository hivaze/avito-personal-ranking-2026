"""R5 / P4 substitute, my own harness: the REAL setup_hzcd of nn_x/w5/cd_arms6c.py (the copy train.py will import) over ALL 246,745,836 train rows, CPU, with a column-lazy stand-in split
(xt[:, c] and xt[ci] are served from the stored feat_cache column with my own attach_xt transform; only the swapped columns are materialised). Captures the log line a GPU run must print
and checks every swapped column of the view tensor against where(keep, stored, fp16(transform(view file))) computed independently in numpy.
usage: r5_setup_full.py <tag>   (HZCD_DIR / HZ2_KEEP from the environment)"""
import os, sys, json, time, types
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, torch
torch.set_num_threads(12)
R = '<repo>'
tag = sys.argv[1]
sys.path.insert(0, f'{R}/nn_x/w5')
import cd_arms6c as m
assert m.__file__ == f'{R}/nn_x/w5/cd_arms6c.py'
FC = f'{R}/nn/feat_cache/train'
CVRAW = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]
CVND = CVRAW + ["ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h",
                "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw",
                "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
NAMES = CVND + ["pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "has:pre2_fe"]
N = 246745836

def my_col(name):
    if name.startswith('has:'):
        x = (~np.isnan(np.load(f'{FC}/{name[4:]}.npy'))).astype(np.float32)
    else:
        x = np.load(f'{FC}/{name}.npy').astype(np.float32)
        if name.endswith('_rk'): pass
        elif name.endswith('_z'): x = np.clip(x, -5, 5)
        elif name.startswith(('seq_cv', 'it_hist')): x = np.log1p(np.maximum(x, 0))
        else: x = np.clip(x, -10, 10)
    return torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16)

class LazyXT:
    dtype, device = torch.float16, torch.device('cpu')
    shape = (N, len(NAMES))
    def __init__(self): self.c = {}
    def col(self, c):
        if c not in self.c: self.c[c] = my_col(NAMES[c])
        return self.c[c]
    def __getitem__(self, k):
        assert isinstance(k, tuple) and k[0] == slice(None) and isinstance(k[1], int), k
        return self.col(k[1])

tr = types.SimpleNamespace(xt=LazyXT(), n_cand=N, device=torch.device('cpu'))
lines = []
a = types.SimpleNamespace(hcap='')
t0 = time.time()
view = m.setup_hzcd(a, tr, NAMES, hold=np.arange(3), t_cool=0, total=10, log=lines.append)
res = {'tag': tag, 'HZ_DIR': m.HZ_DIR, 'KEEP_PATH': m.KEEP_PATH, 'log_line': lines[0], 'cols': [list(c) for c in view.cols], 'sec_setup': time.time() - t0}
# independent check of every swapped column
keep = np.load(m.KEEP_PATH) if m.KEEP_PATH else None
chk = {}
for k, (c, luo_n, hz_n) in enumerate(view.cols):
    x = np.load(f'{m.HZ_DIR}/train/{hz_n}.npy').astype(np.float32)
    if hz_n.endswith('_z'): x = np.clip(x, -5, 5)
    elif hz_n.startswith(('seq_cv', 'it_hist')): x = np.log1p(np.maximum(x, 0))
    elif hz_n.endswith('_has'): pass
    else: x = np.clip(x, -10, 10)
    e = torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16)
    if keep is not None:
        e = torch.where(torch.from_numpy(keep), tr.xt.col(c), e)
    got = view.hz[:, k]
    chk[luo_n] = dict(hz=hz_n, cells_diff=int((got.view(torch.int16) != e.view(torch.int16)).sum()), changed_vs_stored=float((got != tr.xt.col(c)).float().mean()), zero=float((got == 0).float().mean()))
    del x, e
res['check'] = chk
res['all_cells_equal'] = all(v['cells_diff'] == 0 for v in chk.values())
import resource
res['peak_rss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
json.dump(res, open(f'{R}/nn_x/w12/stale/hz2n_review/r5_setup_full_{tag}.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
