"""hz2: CPU fp32 scoring of the 30,000 holdout feeds THROUGH an hz2 view (label-free; writes predictions only). Copy of nn_x/w12/next/poplag/infer.py
(the poplag / poplag_verify inference path, control VALID there) with the view columns taken from <view dir>/train/<hz2 file>.npy at the holdout rows.
Reads poplag's mini split (nn_x/w12/next/poplag/cache/mini, read-only). Lag-0 predictions are poplag's cache/pred_lag0_<member>.npy (same code path).
Usage: CUDA_VISIBLE_DEVICES="" python hold_infer.py <view dir: feats | feats_L> [member,member,... | all] [late | early | full]
  -> cache/hpred_<view dir>_<member>.npy (full) or cache/hpred_<view dir>_<late|early>_<member>.npy (only the holdout feeds of days >= Sep 13 / < Sep 13
     are scored; rows of the other feeds are NaN). Added 03:11Z: at machine load 145 one d192 member on 30,000 feeds takes > 9 min.
"""
import sys
from hz2_common import *
import torch
torch.set_num_threads(12)
sys.path.insert(0, str(ROOT / 'nn_x/w5'))
from data import SplitData  # noqa: E402
from model import Ranker  # noqa: E402

PL = ROOT / 'nn_x/w12/next/poplag/cache'
MINI = PL / 'mini'
W8 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in (951, 952, 953, 954)]
POP1KD = [f'abl_e10_pop1kd05t25_s{s}' for s in (951, 952, 953, 954)]


def ckpt_path(v):
    return ROOT / ('nn_x/w5/ckpt' if v.startswith('w5_') else 'nn_x/abl/ckpt') / f'{v}.pt'


def load_mini():
    L = lambda n: np.load(MINI / f'{n}.npy')
    s = SplitData.__new__(SplitData)
    dev = torch.device('cpu'); g = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    s.split = 'train_holdout'; s.device = dev; s.max_hist = 256
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
    p = g(L('prand').astype(np.float32))
    s.dom = torch.stack([1 - p, p, torch.zeros_like(p)], 1)
    s.crow = L('crow')
    return s


def attach(s, names, vdir):
    cols, used = [], []
    for n in names:
        base = n[4:] if n.startswith('has:') else n
        if base in HZ2_NAME:
            x = np.asarray(np.load(vdir / 'train' / f'{HZ2_NAME[base]}.npy', mmap_mode='r')[s.crow]); used.append(n)
        else:
            x = np.asarray(np.load(FC / 'train' / f'{base}.npy', mmap_mode='r')[s.crow])
        x = (~np.isnan(x)).astype(np.float32) if n.startswith('has:') else xt_transform(n, x)
        cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16))
    s.xt = torch.stack(cols, 1)
    return used


def main():
    vname = sys.argv[1]; vdir = HERE / vname
    mem = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] != 'all' else W8 + POP1KD
    sub = sys.argv[3] if len(sys.argv) > 3 else 'full'
    s = load_mini(); lg(f'mini split loaded ({s.n} feeds, {s.n_cand} rows); view dir {vname}')
    cur = None
    for v in mem:
        out_p = CACHE / (f'hpred_{vname}_{v}.npy' if sub == 'full' else f'hpred_{vname}_{sub}_{v}.npy')
        if out_p.exists():
            lg(f'{v}: exists, skipped'); continue
        ck = torch.load(ckpt_path(v), map_location='cpu', weights_only=False)
        names = ck['cfg']['xt_names']
        if names != cur:
            used = attach(s, names, vdir); cur = names
            lg(f'xt attached ({len(names)} cols); hz2 columns used: {used}')
        m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
        out = torch.full((s.n_cand,), float('nan'), dtype=torch.float32)
        slot = day_of(np.load(MINI / 'ts.npy')) - DAY0
        fsel = np.arange(s.n) if sub == 'full' else np.flatnonzero(slot >= 7 if sub == 'late' else slot < 7)
        t1 = time.time()
        with torch.inference_mode():
            for fb in s.batches(fsel, 65536, 1024, shuffle=False):
                b = s.gather(fb)
                s.scatter(out, b, m(b, use_pos=False))
        np.save(out_p, out.numpy().astype(np.float32))
        lg(f'{v}: {len(fsel)} feeds done in {time.time() - t1:.0f}s -> {out_p.name}')


if __name__ == '__main__':
    main()
