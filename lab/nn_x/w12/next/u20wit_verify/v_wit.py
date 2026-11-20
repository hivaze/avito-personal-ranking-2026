"""u20wit_verify: independent rebuild of the witness sets and an independent CPU fp32 inference path on a random SAMPLE of them.
Own code for: fold hash, holdout draw, witness definition, batch construction straight from the flat memmaps (no SplitData, no sub-split
cache of the study). Only nn_x/w5/model.py (project model class) is imported.
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit_verify/v_wit.py build
  cd <repo> && CUDA_VISIBLE_DEVICES="" .venv/bin/python nn_x/w12/next/u20wit_verify/v_wit.py score <variant> [<variant> ...]
Outputs in nn_x/w12/next/u20wit_verify/cache/: sample.npz, sample_arrays.npz, my_<variant>.npy (logits on the sample's rows, sample order)."""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[k] = '12'
import numpy as np, polars as pl
from pathlib import Path

ROOT = Path('<repo>'); FLAT = ROOT / 'data/flat/train'; COMP = ROOT / 'data/competition/train'
FEAT = ROOT / 'nn_x/w5/feat_cache/train'; CKPT = ROOT / 'nn_x/w5/ckpt'
HERE = ROOT / 'nn_x/w12/next/u20wit_verify'; C = HERE / 'cache'; C.mkdir(exist_ok=True)
STUDY = ROOT / 'nn_x/w12/next/u20wit/cache'
MAXH = 256
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def fold_of(uid, idx):
    u = uid.astype(np.uint64)
    h = ((u * np.uint64(2654435761)) % np.uint64(4294967296)) % np.uint64(5)
    return np.where(uid == 0, idx % 5, h.astype(np.int64))


def meta():
    f = pl.read_parquet(FLAT / 'feeds.parquet', columns=['user_id', 'M', 'H', 'cand_off', 'hist_off'])
    n = f.height; d = {k: f[k].to_numpy().astype(np.int64) for k in ('user_id', 'M', 'H', 'cand_off', 'hist_off')}
    p1 = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz'); assert (p1['fidx'] == np.arange(n)).all()
    pold = np.nan_to_num(p1['p_random'].astype(np.float32), nan=0.0); pold[d['M'] < 50] = 0.0
    cand = np.where((pold > 0.9) & (d['M'] >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    p2 = np.load(ROOT / 'nn_x/w12/xdet/out/prand_v2.npz'); assert (p2['fidx'] == np.arange(n)).all()
    pv2 = np.nan_to_num(p2['p_random'].astype(np.float32), nan=0.0)
    d.update(n=n, pold=pold, pv2=pv2, hold=hold, xf=fold_of(d['user_id'], np.arange(n)))
    return d


if sys.argv[1] == 'build':
    m = meta(); M, xf = m['M'], m['xf']
    inH = np.zeros(m['n'], bool); inH[m['hold']] = True
    ex = (m['pv2'] > 0.9) & ~inH
    W0S = np.where(ex & (xf == 0) & (M < 50))[0]; W0L = np.where(ex & (xf == 0) & (M >= 50))[0]; H0 = m['hold'][xf[m['hold']] == 0]
    XHall = m['hold'][xf[m['hold']] != 0]
    log(f'my sets: W0S {len(W0S)} W0L {len(W0L)} H0 {len(H0)} cross-user holdout {len(XHall)}')
    st = np.load(STUDY / 'sets.npz')
    for k, mine in (('W0S', W0S), ('W0L', W0L), ('H0', H0)):
        log(f'  {k}: study {len(st[k])} mine {len(mine)} identical {bool(len(st[k]) == len(mine) and (st[k] == mine).all())}')
    sxh = np.load(STUDY / 'subxh_meta.npz')['src_fidx']
    log(f'  XH of the study: {len(sxh)} feeds, all in my cross-user holdout {bool(np.isin(sxh, XHall).all())}, any fold-0 {int((xf[sxh] == 0).sum())}')
    # label-free membership facts
    log(f'  fold-0 feeds outside the holdout (u20b unlabelled / u20a dropped): {int(((xf == 0) & ~inH).sum())}')
    pool_c = (m['pold'] > 0.5) & ~inH
    log(f'  cooldown pool: C {int(pool_c.sum())}, A = B {int((pool_c & (xf != 0)).sum())}; W0L in C pool {int(pool_c[W0L].sum())}; W0S in C pool {int(pool_c[W0S].sum())}')
    log(f'  user-0 feeds: W0S {int((m["user_id"][W0S] == 0).sum())} W0L {int((m["user_id"][W0L] == 0).sum())} H0 {int((m["user_id"][H0] == 0).sum())}')
    # users of the witness feeds: do they also own labelled feeds in A's training? (must be none except the user-0 bucket)
    uw = np.unique(m['user_id'][np.concatenate([W0S, W0L])]); uw = uw[uw != 0]
    trainA = (xf != 0) & ~inH
    log(f'  witness users (non-0) {len(uw)}; of them with a feed in A\'s training set: {int(np.isin(uw, np.unique(m["user_id"][trainA])).sum())}')
    rng = np.random.default_rng(424242)
    S = dict(W0S=np.sort(rng.choice(W0S, 2000, replace=False)), W0L=np.sort(rng.choice(W0L, 1500, replace=False)),
             H0=np.sort(rng.choice(H0, 600, replace=False)), XH=np.sort(rng.choice(sxh, 600, replace=False)))
    np.savez(C / 'sample.npz', **S)
    log('sample: ' + ', '.join(f'{k} {len(v)} feeds {int(M[v].sum())} rows' for k, v in S.items()))
    sys.exit(0)

# ---------------------------------------------------------------- score
import torch
torch.set_num_threads(12)
sys.path.insert(0, str(ROOT / 'nn_x/w5'))
from model import Ranker

m = meta(); S = np.load(C / 'sample.npz'); fids = np.concatenate([S[k] for k in ('W0S', 'W0L', 'H0', 'XH')]); nf = len(fids)
Mf = m['M'][fids]; Hf = np.minimum(m['H'][fids], MAXH); co = m['cand_off'][fids]; ho = m['hist_off'][fids]
loc_off = np.concatenate([[0], np.cumsum(Mf)[:-1]]); hloc_off = np.concatenate([[0], np.cumsum(Hf)[:-1]])
rows = np.concatenate([np.arange(o, o + k) for o, k in zip(co, Mf)])
n_hist_total = np.load(FLAT / 'hist_emb_idx.npy', mmap_mode='r').shape[0]
# what data.gather reads at PADDED / EMPTY history positions is the row hist_off (clamped): keep it as one extra "first event" row per feed
hfirst = np.minimum(ho, n_hist_total - 1)
hrows = np.concatenate([np.arange(o, o + k) for o, k in zip(ho, Hf)] + [np.zeros(0, np.int64)])
mm = lambda nme: np.load(FLAT / f'{nme}.npy', mmap_mode='r')
t0 = time.time()
A = {}
for nme in ('cand_emb_idx', 'cand_category_id', 'cand_f1', 'cand_f2', 'cand_f3', 'cand_f4', 'cand_f5'):
    A[nme] = np.asarray(mm(nme)[rows])
for nme in ('hist_emb_idx', 'hist_category_id', 'hist_event_type', 'hist_event_source'):
    A[nme] = np.asarray(mm(nme)[hrows]); A[nme + '@first'] = np.asarray(mm(nme)[hfirst])
uemb = np.asarray(mm('user_emb')[fids])
tmeta = json.loads((COMP / 'item_embeddings/data.meta').read_text())
table = np.memmap(COMP / 'item_embeddings/data.memmap', dtype=np.int8, mode='r', shape=tuple(tmeta['shape']))
need = np.unique(np.concatenate([A['cand_emb_idx'], A['hist_emb_idx'], A['hist_emb_idx@first']]).astype(np.int64)); need = need[need >= 0]
tab = np.asarray(table[need]).astype(np.float32)
lookup = lambda idx: np.where((idx >= 0)[..., None], tab[np.searchsorted(need, np.maximum(idx, 0)).clip(0, len(need) - 1)], 0.0).astype(np.float32)
log(f'sample {nf} feeds, {len(rows)} cand rows, {len(hrows)} hist rows, {len(need)} item rows; raw arrays read in {time.time() - t0:.0f}s')


def load_xt(names):
    cols = []
    for nme in names:
        if nme.startswith('has:'):
            x = (~np.isnan(np.asarray(np.load(FEAT / f'{nme[4:]}.npy', mmap_mode='r')[rows]))).astype(np.float32)
        else:
            x = np.asarray(np.load(FEAT / f'{nme}.npy', mmap_mode='r')[rows]).astype(np.float32)
        if nme.startswith('has:') or nme.endswith('_rk'): pass
        elif nme.endswith('_z'): x = np.clip(x, -5, 5)
        elif nme.startswith(('seq_cv', 'it_hist')): x = np.log1p(np.maximum(x, 0))
        else: x = np.clip(x, -10, 10)
        cols.append(np.nan_to_num(x, nan=0.0).astype(np.float16))
    return np.stack(cols, 1)


def batch(ix, xt):
    """ix = positions in the sample (feeds of similar M). Same tensor layout as data.SplitData.gather, built from my own arrays."""
    B = len(ix); Mb = Mf[ix]; Mx = int(-(-Mb.max() // 16) * 16)
    ar = np.arange(Mx); cmask = ar[None] < Mb[:, None]
    li = np.where(cmask, loc_off[ix][:, None] + ar[None], loc_off[ix][:, None])        # padded slots read the feed's first candidate
    cemb = A['cand_emb_idx'][li]
    Hb = Hf[ix]; Hx = int(Hb.max()); Hx = min(max(-(-Hx // 64) * 64, 64), MAXH)
    ah = np.arange(Hx); hmask = ah[None] < Hb[:, None]
    lh = np.where(hmask, hloc_off[ix][:, None] + ah[None], 0)
    g = lambda nme: np.where(hmask, A[nme][lh] if len(hrows) else 0, A[nme + '@first'][ix][:, None])   # padded slots read the row at hist_off
    hemb = g('hist_emb_idx'); hmiss = (hemb < 0) | ~hmask
    he = lookup(hemb); he[hmiss] = 0.0
    T = torch.from_numpy
    p = m['pold'][fids[ix]].astype(np.float32)
    b = dict(user0=T(m['user_id'][fids[ix]] == 0), fidx=T(fids[ix]), M=T(Mb), H=T(Hb), cmask=T(cmask), hmask=T(hmask),
             ce_raw=T(lookup(cemb)), cmiss=T(cemb < 0), ccat=T(A['cand_category_id'][li].astype(np.int64)),
             cf=T(np.stack([A[f'cand_f{i}'][li] for i in range(1, 6)], -1).astype(np.float32)),
             he_raw=T(he), hemb_ok=T(~hmiss), hcat=T(g('hist_category_id').astype(np.int64)), het=T(g('hist_event_type').astype(np.int64)),
             hes=T(g('hist_event_source').astype(np.int64)), ue_raw=T(uemb[ix].astype(np.float32)),
             hidx=T(hemb.astype(np.int64)),      # the model only de-duplicates on it: any injective id works (global item-table rows here)
             dom=T(np.stack([1 - p, p, np.zeros_like(p)], 1)))
    if xt is not None: b['xt'] = T(xt[li].astype(np.float32))
    if DOM == 'rl': del b['dom']      # val / test convention: no dom in the batch -> model.infer_domain (random-like one-hot)
    return b, cmask, li


order = np.argsort(Mf, kind='stable'); xt_cache = {}
DOM = os.environ.get('DOM', 'tr'); assert DOM in ('tr', 'rl')
for v in sys.argv[2:]:
    outp = C / (f'my_{v}.npy' if DOM == 'tr' else f'my_{v}__rl.npy')
    if outp.exists(): log(f'{v}: exists'); continue
    ck = torch.load(CKPT / f'{v}.pt', map_location='cpu', weights_only=False); cfg = ck['cfg']; a = ck.get('args', {})
    key = json.dumps(cfg.get('xt_names'))
    if key not in xt_cache: xt_cache[key] = load_xt(cfg['xt_names']) if cfg.get('xt_names') else None
    model = Ranker(**cfg); model.load_state_dict(ck['state_dict']); model.eval()
    assert cfg.get('domain'), 'no domain token'
    out = np.full(len(rows), np.nan, np.float32); t0 = time.time(); i = 0
    with torch.no_grad():
        while i < nf:
            mx = int(-(-Mf[order[i]] // 16) * 16); bs = max(1, min(256, 16384 // mx)); j = i
            while j < nf and j - i < bs and int(-(-Mf[order[j]] // 16) * 16) == mx: j += 1
            ix = order[i:j]; b, cmask, li = batch(ix, xt_cache[key])
            lg = model(b, use_pos=False).float().numpy()
            out[li[cmask]] = lg[cmask]; i = j
    assert np.isfinite(out).all()
    np.save(outp, out)
    log(f'{v}: d {cfg["d"]} xfold {a.get("xfold")} unlab {a.get("xfold_unlab")} kd_T {a.get("kd_T")} kd_w {a.get("kd_w")} seed {a.get("seed")} step {ck.get("step", "selected")} | {len(rows)} rows in {time.time() - t0:.0f}s')
log('DONE')
