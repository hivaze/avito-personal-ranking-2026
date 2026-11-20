"""poplag_verify step D: my own CPU fp32 scoring of a random SAMPLE of holdout feeds for one W8 member and one pop1kd member, at lag 0 and on the
private-like v3 columns. LABEL-FREE: predictions are compared with prediction files only (stored bf16, the study's CPU lag-0 replay, the study's
staged private-like predictions). The sample arrays are cut from data/flat by my own code; model / gather are the lead's nn_x/w5 modules (imported).
  cd <repo>/nn_x/w12/next/poplag_verify && CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_infer.py [n_feeds=1000]"""
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


NS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MEM = ['w5_e10_kd05w192_s951', 'abl_e10_pop1kd05t25_s951']
ftr, hold, p = my_holdout()
fM = ftr['M'].to_numpy().astype(np.int64); fH = ftr['H'].to_numpy().astype(np.int64)
fco = ftr['cand_off'].to_numpy().astype(np.int64); fho = ftr['hist_off'].to_numpy().astype(np.int64)
pick = np.sort(np.random.default_rng(951115).choice(len(hold), NS, replace=False)); fx = hold[pick]
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
pp = g(p[fx].astype(np.float32)); s.dom = torch.stack([1 - pp, pp, torch.zeros_like(pp)], 1)       # nn_x/w5/train.py l.372-373
lg(f'sample of {NS} holdout feeds: {len(crow)} rows, {len(hrow)} history events, {len(used)} table rows')
LAGC = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo', 'xp_imp_c50']


def xt_for(names, view):
    cols = []; rep = []
    for n in names:
        base = n[4:] if n.startswith('has:') else n
        if view != 'lag0' and base in LAGC:
            src = MYC / 'my_priv_v3_xp_imp_c50.npy' if base == 'xp_imp_c50' else STUDY / f'lag_{view}_{base}.npy'   # xp: MY rebuilt column
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


Ms = s.M_np; os_ = s.cand_off_np; fid = np.repeat(np.arange(NS), Ms)


def zf(x):
    return zfeed(x, os_, Ms).astype(np.float64)


def top1(x):
    o = np.lexsort((-x, fid)); return o[os_]


for v in MEM:
    ck = torch.load(ROOT / ('nn_x/w5/ckpt' if v.startswith('w5_') else 'nn_x/abl/ckpt') / f'{v}.pt', map_location='cpu', weights_only=False)
    a = ck['args']
    lg(f'{v}: args domain={a.get("domain")} holdout={a.get("holdout")} prand2={a.get("prand2")} kd_w={a.get("kd_w")} d={ck["cfg"]["d"]} xt cols {len(ck["cfg"]["xt_names"])}')
    m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
    out = {}
    for view in ('lag0', 'priv_v3'):
        s.xt, rep = xt_for(ck['cfg']['xt_names'], view)
        o = torch.zeros(s.n_cand, dtype=torch.float32)
        with torch.inference_mode():
            for fb in s.batches(np.arange(s.n), 65536, 1024, shuffle=False):
                b = s.gather(fb); s.scatter(o, b, m(b, use_pos=False))
        out[view] = o.numpy().astype(np.float64)
        lg(f'  {view}: scored; columns replaced {rep}')
    st = np.load(stored_path(v))[hpos].astype(np.float64); c0 = np.load(STUDY / f'pred_lag0_{v}.npy')[hpos].astype(np.float64); c1 = np.load(STUDY / f'pred_priv_v3_{v}.npy')[hpos].astype(np.float64)
    for nm, mine, ref in (('MY lag-0 vs the study\'s CPU lag-0 file', out['lag0'], c0), ('MY lag-0 vs the STORED bf16 file', out['lag0'], st),
                          ('MY private-like v3 vs the study\'s STAGED private-like file', out['priv_v3'], c1), ('(contrast) MY private-like v3 vs MY lag-0', out['priv_v3'], out['lag0'])):
        fc = np.add.reduceat(zf(mine) * zf(ref), os_) / Ms
        print(f'   {nm}: max |diff| {np.abs(mine - ref).max():.2e}, mean |diff| {np.abs(mine - ref).mean():.2e}, mean per-feed corr {fc.mean():.6f}, top-1 changed in {np.mean(top1(mine) != top1(ref)):.4f} of {NS} feeds', flush=True)
lg('done')
