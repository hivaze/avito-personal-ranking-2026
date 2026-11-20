"""poplag_verify step E (extras, LABEL-FREE): my own CPU fp32 scoring of a PART of the 30,000 holdout feeds for all 12 members on a lagged view.
Used for the public-like view, which the study never scored. The part is 'late' (holdout days >= Sep 13) or 'early' (the rest).
xt columns: stored nn/feat_cache/train columns, with the recency-dependent ones replaced by the lagged columns
(it_hist_* / seq_cv_*: the study's cache/lag_<view>_<col>.npy, checked on a sample by v_sample.py; xp_imp_c50: MY full rebuild cache/my_<pub|priv>_v3_xp_imp_c50.npy).
Output: cache/pred_<view>_<part>_<member>.npy (float32, rows of the part in holdout row order). Resumable (existing files are skipped).
  cd <repo>/nn_x/w12/next/poplag_verify && CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_infer_full.py pub_v3 late"""
from v_common import *
import time, json
import torch
torch.set_num_threads(12)
sys.path.insert(0, str(ROOT / 'nn_x/w5'))
from data import SplitData  # noqa: E402
from model import Ranker  # noqa: E402

T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


VIEW = sys.argv[1]; PART = sys.argv[2]
MEM = sys.argv[3].split(',') if len(sys.argv) > 3 else W8 + P4
assert VIEW in ('pub_v3', 'priv_v3', 'lag0') and PART in ('late', 'early')
ftr, hold, p = my_holdout()
fM = ftr['M'].to_numpy().astype(np.int64); fH = ftr['H'].to_numpy().astype(np.int64)
fco = ftr['cand_off'].to_numpy().astype(np.int64); fho = ftr['hist_off'].to_numpy().astype(np.int64)
hday = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY
late = hday >= SEP6 + 7
pick = np.flatnonzero(late if PART == 'late' else ~late); fx = hold[pick]; NS = len(pick)
off_h = np.concatenate([[0], np.cumsum(fM[hold])[:-1]])
hpos = np.concatenate([np.arange(off_h[i], off_h[i] + fM[hold[i]]) for i in pick])
crow = np.concatenate([np.arange(fco[f], fco[f] + fM[f]) for f in fx]); hrow = np.concatenate([np.arange(fho[f], fho[f] + fH[f]) for f in fx])
mm = lambda n: np.load(FLAT / 'train' / f'{n}.npy', mmap_mode='r')
g = lambda a: torch.from_numpy(np.ascontiguousarray(a))
ce = np.asarray(mm('cand_emb_idx')[crow]); he = np.asarray(mm('hist_emb_idx')[hrow])
d = ROOT / 'data/competition/train/item_embeddings'; shape = tuple(json.loads((d / 'data.meta').read_text())['shape'])
tab = np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=shape)
used = np.unique(np.concatenate([ce[ce >= 0], he[he >= 0]])); pos = np.full(int(used.max()) + 1, -1, np.int64); pos[used] = np.arange(len(used))
s = SplitData.__new__(SplitData)
s.split = 'verify_part'; s.device = torch.device('cpu'); s.max_hist = 256
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
pp = g(p[fx].astype(np.float32)); s.dom = torch.stack([1 - pp, pp, torch.zeros_like(pp)], 1)       # nn_x/w5/train.py l.372-373
lg(f'{VIEW} / {PART}: {NS} holdout feeds, {len(crow)} rows, {len(hrow)} history events, {len(used)} table rows')
LAGC = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'xp_imp_c50']


def xt_for(names):
    cols = []; rep = []
    for n in names:
        base = n[4:] if n.startswith('has:') else n
        if VIEW != 'lag0' and base in LAGC:
            src = MYC / f'my_{VIEW}_xp_imp_c50.npy' if base == 'xp_imp_c50' else STUDY / f'lag_{VIEW}_{base}.npy'
            x = np.load(src, mmap_mode='r')[hpos].astype(np.float32); rep.append(base)
        else:
            x = np.asarray(np.load(FC / 'train' / f'{base}.npy', mmap_mode='r')[crow]).astype(np.float32)
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
    return torch.stack(cols, 1), rep


for v in MEM:
    outp = MYC / f'pred_{VIEW}_{PART}_{v}.npy'
    if outp.exists():
        lg(f'{v}: exists, skipped'); continue
    ck = torch.load(ROOT / ('nn_x/w5/ckpt' if v.startswith('w5_') else 'nn_x/abl/ckpt') / f'{v}.pt', map_location='cpu', weights_only=False)
    m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
    s.xt, rep = xt_for(ck['cfg']['xt_names'])
    o = torch.zeros(s.n_cand, dtype=torch.float32)
    with torch.inference_mode():
        for fb in s.batches(np.arange(s.n), 65536, 1024, shuffle=False):
            b = s.gather(fb); s.scatter(o, b, m(b, use_pos=False))
    tmp = MYC / f'tmp_{VIEW}_{PART}_{v}.npy'
    np.save(tmp, o.numpy()); os.replace(tmp, outp)
    lg(f'{v}: scored ({len(rep)} columns replaced: {rep}); saved {outp.name}')
lg('ALL DONE')
