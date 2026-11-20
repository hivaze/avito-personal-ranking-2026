"""hz2n P4a: the REAL cd_arms6c.setup_hzcd (unwrapped load_hz, the real HZ2_KEEP file) over the FULL train row space (246,745,836 rows) on a CPU stand-in split that
holds only the stored fp16 columns of the swappable inputs (the pattern of hz2_review/dropin_own.py). For each directory: the log line; every cell of every swapped
column == keep ? stored : fp16(attach_xt transform(view file)); encoder columns: no-ID value on newly cold rows, stored bits elsewhere; a gather of 500 random feeds
through HzTrainView returns exactly those cells. Peak memory about 20 GB.   python p4a_full.py [dirs...]  -> logs/p4a_full.log, p4a_full.json"""
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True
assert os.environ.get('CUDA_VISIBLE_DEVICES', 'x') == ''
import numpy as np
import polars as pl
import torch

torch.set_num_threads(12)
import test_cd_arms6c as T
from mini_split import _xt_col

HERE, ROOT, HZ2 = T.HERE, T.ROOT, T.HZ2
N = 246745836
SUF = ('_' + '_'.join(sys.argv[1:])) if sys.argv[1:] else ''
NAMES = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo',
         'pre2_fe_z', 'pre2_new_z', 'pre2_fe_s', 'pre2_new_s']
t0 = time.time()
out = []


def P(*a):
    s = f'[{time.time() - t0:5.0f}s] ' + ' '.join(str(x) for x in a)
    out.append(s); print(s, flush=True)
    open(f'{HERE}/logs/p4a_full{SUF}.log', 'w').write('\n'.join(out) + '\n')


class Full:
    def __init__(self):
        self.split, self.device = 'train', torch.device('cpu')
        self.xt = torch.empty((N, len(NAMES)), dtype=torch.float16)
        for j, n in enumerate(NAMES):
            self.xt[:, j] = _xt_col('train', n, slice(None))   # mini_split._xt_col = data.SplitData.attach_xt's transform, fp16
        self.n_cand = N
        f = pl.read_parquet(f'{ROOT}/data/flat/train/feeds.parquet')
        self.M_np = f['M'].to_numpy().astype(np.int64); self.cand_off_np = f['cand_off'].to_numpy().astype(np.int64)
        self.n = len(f)

    def gather(self, fidx_np, with_pos=False):   # the ci / cmask / xt part of data.SplitData.gather
        M = torch.from_numpy(self.M_np[fidx_np])
        Mx = int(-(-self.M_np[fidx_np].max() // 16) * 16)
        ar = torch.arange(Mx)
        cmask = ar[None, :] < M[:, None]
        off = torch.from_numpy(self.cand_off_np[fidx_np])
        ci = torch.where(cmask, off[:, None] + ar[None, :], off[:, None])
        return dict(ci=ci, cmask=cmask, xt=self.xt[ci].float())


def i16(t):
    return t.contiguous().view(torch.int16)


tr = Full()
P(f'stand-in split: {N} rows x {len(NAMES)} stored columns {NAMES} ({tr.xt.numel() * 2 / 2**30:.2f} GB)')
S_cold_stored = None
res = {}
dirs = sys.argv[1:] or ['feats_n', 'feats_Ln', 'feats_hn', 'feats_nF', 'feats_LnF', 'feats_hnF', 'feats_nP', 'feats_LnP', 'feats_hnP']
feeds = np.sort(np.random.default_rng(99).choice(tr.n, 500, replace=False))
for dn in dirs:
    base, has_keep = T.BASES[dn.rstrip('PFS')]
    d = f'{HERE}/{dn}'
    env = {'HZCD_DIR': d}
    if has_keep:
        env['HZ2_KEEP'] = f'{base}/train/hz2_keep_rows.npy'
    C = T.load_mod(T.M6C, 'full_' + dn, env)
    logs = []
    view = C.setup_hzcd(types.SimpleNamespace(hcap='', variant='p4a'), tr, NAMES, np.arange(5), 10, 100, logs.append)
    names = [c[1] for c in view.cols]
    keep = torch.from_numpy(np.load(env['HZ2_KEEP'])) if has_keep else torch.zeros(N, dtype=torch.bool)
    nc = torch.from_numpy(np.load(f'{d}/train/hz2n_newly_cold.npy'))
    r = {'columns': names, 'log': logs[-1], 'cells_not_equal_expected': {}, 'rows_view_ne_stored': {}, 'hz_GB': view.hz.numel() * 2 / 2**30}
    for k, (c, sn, hn) in enumerate(view.cols):
        exp = torch.from_numpy(C.xt_transform(hn, np.load(f'{d}/train/{hn}.npy'))).to(torch.float16)
        exp = torch.where(keep, tr.xt[:, c], exp)
        r['cells_not_equal_expected'][sn] = int((i16(view.hz[:, k]) != i16(exp)).sum())
        r['rows_view_ne_stored'][sn] = int((i16(view.hz[:, k]) != i16(tr.xt[:, c])).sum())
        if sn in ('pre2_fe_s', 'pre2_new_s'):
            r[f'{sn}_rows_changed_outside_newly_cold'] = int(((i16(view.hz[:, k]) != i16(tr.xt[:, c])) & ~nc).sum())
            src = np.load(f'{ROOT}/data/feats/train/pre2_fe_noid.npy') if (sn == 'pre2_fe_s' and not dn.endswith('P')) else np.load(f'{d}/train/{"pre2_hz_fe" if sn == "pre2_fe_s" else "pre2_hz_new"}.npy')
            e2 = torch.from_numpy(np.nan_to_num(np.clip(((src + 20) / 6).astype(np.float32), -10, 10), nan=0.0)).to(torch.float16)
            r[f'{sn}_newly_cold_rows_not_the_no_id_value'] = int(((i16(view.hz[:, k]) != i16(e2)) & nc).sum())
            del src, e2
        del exp
    b = view.gather(feeds, with_pos=True)
    cidx = [c[0] for c in view.cols]
    want = view.hz[b['ci']].float()
    r['gather_500_feeds_swapped_cells_equal'] = bool(torch.equal(b['xt'][..., cidx].contiguous().view(-1).view(torch.int32), want.contiguous().view(-1).view(torch.int32)))
    other = [j for j in range(len(NAMES)) if j not in cidx]
    r['gather_500_feeds_other_columns_equal_plain'] = bool(torch.equal(b['xt'][..., other], tr.gather(feeds)['xt'][..., other])) if other else True
    r['gather_newly_cold_cells'] = int((nc[b['ci']] & b['cmask']).sum())
    r['pass'] = bool(sum(r['cells_not_equal_expected'].values()) == 0 and r['gather_500_feeds_swapped_cells_equal'] and r['gather_500_feeds_other_columns_equal_plain']
                     and all(v == 0 for k_, v in r.items() if k_.endswith('_outside_newly_cold') or k_.endswith('_not_the_no_id_value')))
    res[dn] = r
    P(f'P4a {"PASS" if r["pass"] else "FAIL"} [{dn}]: {len(names)} swapped columns; hz tensor {r["hz_GB"]:.2f} GB; cells != keep ? stored : fp16(transform(file)) over all {N} rows: '
      f'{r["cells_not_equal_expected"]}; rows where the view differs from the stored column: {r["rows_view_ne_stored"]}; '
      + '; '.join(f'{k_}: {v}' for k_, v in r.items() if k_.endswith('_outside_newly_cold') or k_.endswith('_not_the_no_id_value'))
      + f'; gather of 500 random feeds: swapped cells == view tensor {r["gather_500_feeds_swapped_cells_equal"]}, other columns == plain {r["gather_500_feeds_other_columns_equal_plain"]}, '
      f'newly cold cells in it {r["gather_newly_cold_cells"]}')
    P('   log: ' + logs[-1])
    json.dump(res, open(f'{HERE}/p4a_full{SUF}.json', 'w'), indent=1)
    del view, keep, nc, b, want
P('P4a DONE')
