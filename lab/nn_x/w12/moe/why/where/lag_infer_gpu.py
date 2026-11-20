"""why/where W2: GPU re-scoring of reference and MoE-screen checkpoints on the 30,000 holdout feeds with popularity inputs at
lag 0 (stored) and at the test lag (poplag views priv_v3 / pub_v3, built by lab/nn_x/w12/next/poplag/build_lag2.py + v3).
Read-only on everything outside lab/nn_x/w12/moe/why/where/. Copy of the poplag/infer.py data path (mini split, attach_xt
transform, dom = (1-p, p, 0)), but on ONE GPU with the stored files' numerics (common.predict: bf16 autocast, tf32 on) and the
w12/moe model module (a superset of nn_x/w5/model.py: flag-off checkpoints build and run the w5 forward exactly).

Usage: CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 .venv/bin/python lab/nn_x/w12/moe/why/where/lag_infer_gpu.py <view,view> <member> ...
  view = lag0 | priv_v3 | pub_v3 ; member = w5_e10_<arm>_s<seed>[@last]
Writes out/lag/pred_<view>_<member>.npy (float32, holdout row order) and appends a line per member to logs/lag_infer.log,
with the lag-0 control (NDCG of the stored holdout preds vs this path; correlation)."""
import os, sys, time, json
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '12'
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(12)
ROOT = Path('<repo>')
sys.path.insert(0, str(ROOT / 'lab/nn_x/w5'))
sys.path.insert(0, str(ROOT / 'lab/nn_x/w12/moe'))
sys.path.insert(1, str(ROOT / 'src'))
from data import SplitData  # noqa: E402
from model_moe import Ranker  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

PL = ROOT / 'lab/nn_x/w12/next/poplag/cache'
MINI = PL / 'mini'
FC = ROOT / 'nn/feat_cache'
HERE = ROOT / 'lab/nn_x/w12/moe/why/where'
OUT = HERE / 'out/lag'
LAGCOLS = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo',
           'it_hist_et3_luo', 'it_hist_et4_luo', 'xp_imp_c50']
T0 = time.time()
lg = lambda *a: print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def ck_path(v):
    for d in (ROOT / 'lab/nn_x/w12/moe/out/ckpt', ROOT / 'nn_x/w5/ckpt'):
        if (d / f'{v}.pt').exists():
            return d / f'{v}.pt'
    raise FileNotFoundError(v)


def stored_holdout(v):
    for d in (ROOT / 'lab/nn_x/w12/moe/out/preds/holdout', ROOT / 'nn_x/w5/preds/holdout'):
        if (d / f'nn_{v}_holdout.npy').exists():
            return d / f'nn_{v}_holdout.npy'
    return None


def xt_transform(name, x):
    """poplag/pl_common.xt_transform = nn_x/w5/data.py attach_xt transform."""
    x = np.asarray(x).astype(np.float32)
    if name.startswith('has:') or name.endswith('_rk'):
        pass
    elif name.endswith('_z'):
        x = np.clip(x, -5, 5)
    elif name.startswith(('seq_cv', 'it_hist')):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


def load_mini(dev):
    L = lambda n: np.load(MINI / f'{n}.npy')
    s = SplitData.__new__(SplitData)
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    s.split = 'train_holdout'; s.device = dev; s.max_hist = 256; s.hist_ext = 0
    s.M_np = L('M'); s.H_np = L('H'); s.n = len(s.M_np)
    s.cand_off_np = np.concatenate([[0], np.cumsum(s.M_np)[:-1]]).astype(np.int64)
    s.user_id = L('user_id'); s.dom = None
    s.M = g(s.M_np); s.H = g(s.H_np); s.cand_off = g(s.cand_off_np)
    s.hist_off = g(np.concatenate([[0], np.cumsum(s.H_np)[:-1]]).astype(np.int64))
    s.user_emb = g(L('user_emb')); s.user0 = g(s.user_id == 0)
    s.c_emb = g(L('c_emb')); s.c_item = None; s.h_item = None; s.xt = None; s.n_cand = len(s.c_emb)
    s.c_cat = g(L('c_cat')); s.c_f = g(L('c_f'))
    s.has_target = False; s.target_np = None; s.c_tgt = None; s.has_pos = False; s.c_pos = None
    s.h_emb = g(L('h_emb')); s.h_cat = g(L('h_cat')); s.h_et = g(L('h_et')); s.h_es = g(L('h_es')); s.n_hist = len(s.h_emb)
    s.table = g(L('table'))
    p = torch.from_numpy(L('prand').astype(np.float32))
    s.dom = torch.stack([1 - p, p, torch.zeros_like(p)], 1).to(dev)
    s.crow = L('crow')
    return s


def raw_col(name, crow, view):
    p = PL / f'lag_{view}_{name}.npy'
    if view != 'lag0' and p.exists():
        x = np.load(p); assert len(x) == len(crow), (name, len(x))
        return x, True
    return np.asarray(np.load(FC / 'train' / f'{name}.npy', mmap_mode='r')[crow]), False


def attach(s, names, view):
    cols, used = [], []
    for n in names:
        if n.startswith('has:'):
            x, u = raw_col(n[4:], s.crow, view); x = (~np.isnan(x)).astype(np.float32)
        else:
            x, u = raw_col(n, s.crow, view); x = xt_transform(n, x)
        if u: used.append(n)
        cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16))
    s.xt = torch.stack(cols, 1).to(s.device)
    return used


@torch.no_grad()
def predict(m, s):
    m.eval()
    out = torch.zeros(s.n_cand, dtype=torch.float32, device=s.device)
    for fb in s.batches(np.arange(s.n), 65536, 1024, shuffle=False):
        b = s.gather(fb)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lgt = m(b, use_pos=False)
        s.scatter(out, b, lgt)
    return out.cpu().numpy()


def main():
    views = sys.argv[1].split(','); mem = sys.argv[2:]
    dev = torch.device('cuda:0'); torch.backends.cuda.matmul.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_mini(dev); lg(f'mini loaded: {s.n} feeds {s.n_cand} rows on {dev}')
    Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
    yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
    log = open(HERE / 'logs/lag_infer.log', 'a')
    cur = None
    for view in views:
        for v in mem:
            op = OUT / f'pred_{view}_{v}.npy'
            if op.exists():
                lg(f'{view} {v}: exists'); continue
            t1 = time.time()
            ck = torch.load(ck_path(v), map_location='cpu', weights_only=False)
            names = ck['cfg']['xt_names']
            if (names, view) != cur:
                used = attach(s, names, view); cur = (names, view)
                exp = [n for n in names if n in LAGCOLS] if view != 'lag0' else []
                assert sorted(used) == sorted(exp), (view, used, exp)
                lg(f'xt attached view {view}: rebuilt {used}')
            m = Ranker(**ck['cfg']).to(dev); m.load_state_dict(ck['state_dict']); m.eval()
            pr = predict(m, s)
            np.save(op, pr.astype(np.float32))
            nd = ndcg_per_feed(pr, yh, oh, Mh).mean()
            msg = ''
            if view == 'lag0':
                sp = stored_holdout(v)
                if sp is not None:
                    r = np.load(sp).astype(np.float32)
                    msg = f' | stored {ndcg_per_feed(r, yh, oh, Mh).mean():.5f} corr {np.corrcoef(r, pr)[0, 1]:.6f} max|d| {np.abs(r - pr).max():.3e}'
            line = f'{time.strftime("%FT%TZ", time.gmtime())} {view} {v}: holdout ndcg {nd:.5f}{msg} ({time.time() - t1:.0f}s)'
            lg(line); log.write(line + '\n'); log.flush()
            del m; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
