"""poplag_verify step L (LABEL-FREE): all 12 members re-scored by me on the 400 sampled SUBSET feeds with MY rebuilt private-like columns
(cache/sample_priv_<col>.npy written by v_sample.py), compared with the study's staged private-like predictions and its CPU lag-0 replay.
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_sample12.py"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *
import time, json
import torch
torch.set_num_threads(int(NT))
sys.path.insert(0, str(ROOT / 'nn_x/w5'))
from data import SplitData  # noqa: E402
from model import Ranker  # noqa: E402

T0 = time.time()
OUT = []


def lg(*a):
    s = f'[{time.time() - T0:5.0f}s] ' + ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)


ftr, hold, prand = my_holdout()
fM = ftr['M'].to_numpy().astype(np.int64); fH = ftr['H'].to_numpy().astype(np.int64)
fco = ftr['cand_off'].to_numpy().astype(np.int64); fho = ftr['hist_off'].to_numpy().astype(np.int64)
pick = np.load(MYC / 'sample_pick.npy'); hpos = np.load(MYC / 'sample_hpos.npy'); fx = hold[pick]; NS = len(pick)
crow = np.concatenate([np.arange(fco[f], fco[f] + fM[f]) for f in fx]); hrow = np.concatenate([np.arange(fho[f], fho[f] + fH[f]) for f in fx])
assert len(crow) == len(hpos)
LAGC = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'xp_imp_c50']
MY = {c: np.load(MYC / f'sample_priv_{c}.npy') for c in LAGC}
mm = lambda n: np.load(FLAT / 'train' / f'{n}.npy', mmap_mode='r')
g = lambda a: torch.from_numpy(np.ascontiguousarray(a))
ce = np.asarray(mm('cand_emb_idx')[crow]); he = np.asarray(mm('hist_emb_idx')[hrow])
d = ROOT / 'data/competition/train/item_embeddings'; shape = tuple(json.loads((d / 'data.meta').read_text())['shape'])
tab = np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=shape)
used = np.unique(np.concatenate([ce[ce >= 0], he[he >= 0]])); pos = np.full(int(used.max()) + 1, -1, np.int64); pos[used] = np.arange(len(used))
s = SplitData.__new__(SplitData)
s.split = 'verify_sample'; s.device = torch.device('cpu'); s.max_hist = 256
s.M_np = fM[fx]; s.H_np = fH[fx]; s.n = NS
s.cand_off_np = np.concatenate([[0], np.cumsum(s.M_np)[:-1]]).astype(np.int64)
s.user_id = ftr['user_id'].to_numpy()[fx]
s.M = g(s.M_np); s.H = g(s.H_np); s.cand_off = g(s.cand_off_np); s.hist_off = g(np.concatenate([[0], np.cumsum(s.H_np)[:-1]]).astype(np.int64))
s.user_emb = g(np.asarray(mm('user_emb')[fx])); s.user0 = g(s.user_id == 0)
s.c_emb = g(np.where(ce >= 0, pos[np.maximum(ce, 0)], -1).astype(np.int32)); s.h_emb = g(np.where(he >= 0, pos[np.maximum(he, 0)], -1).astype(np.int32))
s.table = g(np.asarray(tab[used])); s.n_cand = len(crow); s.n_hist = len(hrow)
s.c_cat = g(np.asarray(mm('cand_category_id')[crow]).astype(np.int16)); s.c_f = g(np.stack([np.asarray(mm(f'cand_f{i}')[crow]) for i in range(1, 6)], 1))
s.h_cat = g(np.asarray(mm('hist_category_id')[hrow]).astype(np.int16)); s.h_et = g(np.asarray(mm('hist_event_type')[hrow])); s.h_es = g(np.asarray(mm('hist_event_source')[hrow]))
s.c_item = None; s.h_item = None; s.xt = None; s.has_target = False; s.target_np = None; s.c_tgt = None; s.has_pos = False; s.c_pos = None
pp = g(prand[fx].astype(np.float32)); s.dom = torch.stack([1 - pp, pp, torch.zeros_like(pp)], 1)


def xt_for(names):
    cols = []
    for n in names:
        base = n[4:] if n.startswith('has:') else n
        x = MY[base].astype(np.float32) if base in LAGC else np.asarray(np.load(FC / 'train' / f'{base}.npy', mmap_mode='r')[crow]).astype(np.float32)
        if n.startswith('has:'):
            x = (~np.isnan(x)).astype(np.float32)
        elif n.endswith('_rk'):
            pass
        elif n.endswith('_z'):
            x = np.clip(x, -5, 5)
        elif n.startswith(('seq_cv', 'it_hist')):
            x = np.log1p(np.maximum(x, 0))
        else:
            x = np.clip(x, -10, 10)
        cols.append(torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(torch.float16))
    return torch.stack(cols, 1)


Ms = s.M_np; os_ = s.cand_off_np; fid = np.repeat(np.arange(NS), Ms)
top1 = lambda x: np.lexsort((-x, fid))[os_]
worst = 0.0
lg(f'{NS} sampled SUBSET feeds, {len(crow)} rows; all nine lag-dependent columns are MY rebuild')
for v in W8 + P4:
    ck = torch.load(ROOT / ('nn_x/w5/ckpt' if v.startswith('w5_') else 'nn_x/abl/ckpt') / f'{v}.pt', map_location='cpu', weights_only=False)
    m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
    s.xt = xt_for(ck['cfg']['xt_names'])
    o_ = torch.zeros(s.n_cand, dtype=torch.float32)
    with torch.inference_mode():
        for fb in s.batches(np.arange(s.n), 65536, 1024, shuffle=False):
            b = s.gather(fb); s.scatter(o_, b, m(b, use_pos=False))
    mine = o_.numpy().astype(np.float64); ref = np.load(STUDY / f'pred_priv_v3_{v}.npy', mmap_mode='r')[hpos].astype(np.float64)
    l0 = np.load(STUDY / f'pred_lag0_{v}.npy', mmap_mode='r')[hpos].astype(np.float64)
    worst = max(worst, np.abs(mine - ref).max())
    lg(f'{v:28s} MY private-like vs STAGED: max |diff| {np.abs(mine - ref).max():.2e}, top-1 changed in {np.mean(top1(mine) != top1(ref)):.4f} of feeds | (contrast) staged vs lag-0 file: mean |diff| {np.abs(ref - l0).mean():.3f}, top-1 changed {np.mean(top1(ref) != top1(l0)):.3f}')
lg(f'largest |mine - staged| over the 12 members: {worst:.2e}')
(HERE / 'logs' / 'v_sample12.txt').write_text('\n'.join(OUT) + '\n')
