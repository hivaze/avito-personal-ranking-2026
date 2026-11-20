"""R1c worker: one module (6b or 6c) in one environment; my own stand-in split on my own random feeds. Prints a JSON of sha256 hashes.
usage: r1c_one.py <module_dir> <module_name> <out.json>   (HZCD_DIR / HZ2_KEEP from the environment)"""
import os, sys, json, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np, torch, polars as pl
torch.set_num_threads(12)
R = '<repo>'
mod_dir, mod_name, out = sys.argv[1:4]
sys.path.insert(0, mod_dir)
m = __import__(mod_name)
assert os.path.dirname(os.path.abspath(m.__file__)) == os.path.abspath(mod_dir), m.__file__
FC = f'{R}/nn/feat_cache/train'
CVRAW = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]
CVND = CVRAW + ["ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h",
                "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw",
                "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
PRE2S = CVND + ["pre2_fe_z", "pre2_new_z", "pre2_fe_s", "pre2_new_s", "pre2_hmax32", "has:pre2_fe"]
POPL = [n for n in PRE2S if not n.startswith(("seq_cv", "it_hist"))] + ["xp_imp_c50"]
XT = {'cvnd_pre2s': PRE2S, 'poplight': POPL}

f = pl.read_parquet(f'{R}/data/flat/train/feeds.parquet')
M_all = f['M'].to_numpy().astype(np.int64); off_all = f['cand_off'].to_numpy().astype(np.int64)
ts = f['timestamp'].to_numpy(); slot = (ts + 3 * 3600) // 86400 - 20702
rng = np.random.default_rng(20261115)
# 2,400 feeds: 160 per day slot where available (covers keep days Sep 6-8 and every encoder window)
feeds = np.sort(np.concatenate([rng.choice(np.flatnonzero(slot == s), min(200, int((slot == s).sum())), replace=False) for s in range(15) if (slot == s).sum() > 0]))
M = M_all[feeds]; off = np.concatenate([[0], np.cumsum(M)[:-1]])
rows = np.concatenate([np.arange(o, o + mm) for o, mm in zip(off_all[feeds], M)])
n = len(rows)

def my_attach(name):  # my own copy of the attach_xt rule (data.py lines 180-198), written from its docstring
    if name.startswith('has:'):
        return (~np.isnan(np.load(f'{FC}/{name[4:]}.npy', mmap_mode='r')[rows])).astype(np.float32)
    x = np.load(f'{FC}/{name}.npy', mmap_mode='r')[rows].astype(np.float32)
    if name.endswith('_rk'):
        pass
    elif name.endswith('_z'):
        x = np.clip(x, -5, 5)
    elif name.startswith(('seq_cv', 'it_hist')):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)

class Stand:
    """compact-row stand-in of SplitData: xt [n, F] fp16, gather(fidx) -> dict(ci, cmask, xt, M, fidx)"""
    def __init__(self, names):
        self.xt = torch.stack([torch.from_numpy(my_attach(nm)).to(torch.float16) for nm in names], 1)
        self.n_cand, self.device = n, self.xt.device
    def gather(self, fidx_np, with_pos=False):
        Mb = torch.from_numpy(M[fidx_np]); Mx = int(Mb.max())
        ar = torch.arange(Mx)[None, :]
        cmask = ar < Mb[:, None]
        ci = torch.from_numpy(off[fidx_np])[:, None] + torch.where(cmask, ar, torch.zeros_like(ar))
        return dict(fidx=torch.from_numpy(fidx_np), M=Mb, cmask=cmask, ci=ci, xt=self.xt[ci].float())

def h(t):
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes() + str(a.shape).encode() + str(a.dtype).encode()).hexdigest()

res = {'module': m.__file__, 'HZ_DIR': m.HZ_DIR, 'KEEP_PATH': m.KEEP_PATH, 'n_feeds': len(feeds), 'n_rows': int(n)}
for tag, names in XT.items():
    r = {}
    try:
        cols = m.luo_cols(names)
    except AssertionError as e:
        r['luo_cols_assert'] = str(e)[:80]
        res[tag] = r
        continue
    r['cols'] = [list(c) for c in cols]
    tr = Stand(names)
    hz = m.load_hz('train', cols, tr.device, rows=rows)
    r['hz_raw'] = h(hz); r['hz_shape'] = list(hz.shape)
    nk = 0
    if m.KEEP_PATH:
        keep = torch.from_numpy(np.load(m.KEEP_PATH, mmap_mode='r')[rows].copy())
        nk = int(keep.sum())
        hz = m.apply_keep(hz, tr.xt, cols, keep)
    r['n_keep'] = nk; r['hz_keep'] = h(hz)
    view = m.HzTrainView(tr, cols, hz)
    r['hz_stats'] = m.hz_stats(tr, view)
    g = np.random.default_rng(7)
    order = g.permutation(len(feeds))
    bh = hashlib.sha256(); nb = 0; ncell = 0; changed = 0
    for i in range(0, len(order), 64):
        fb = np.sort(order[i:i + 64])
        b = view.gather(fb)
        b0 = tr.gather(fb)
        for k in sorted(b):
            bh.update(k.encode()); bh.update(h(b[k]).encode())
        ncell += int(b['cmask'].sum()); nb += 1
        changed += int((b['xt'] != b0['xt']).any(-1)[b['cmask']].sum())
        # columns not swapped must equal the plain gather
        ns = [c for c in range(len(names)) if c not in view.col_idx]
        assert torch.equal(b['xt'][..., ns], b0['xt'][..., ns])
        # swapped columns equal the hz tensor
        assert torch.equal(b['xt'][..., view.col_idx], hz[b['ci']].float())
    r['batches'] = nb; r['cells'] = ncell; r['batch_hash'] = bh.hexdigest(); r['cells_changed_vs_plain'] = changed
    res[tag] = r
json.dump(res, open(out, 'w'), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'hz_stats'}) for k, v in res.items()}))
