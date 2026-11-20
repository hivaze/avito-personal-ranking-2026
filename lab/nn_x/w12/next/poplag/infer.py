"""poplag: ONE CPU fp32 inference path for the 30K holdout feeds (label-free; writes predictions only).

A mini split (SplitData attribute layout, nn_x/w5/data.py, restricted to the 30,000 holdout feeds in train order) is built once into
cache/mini/ ; batches / gather / scatter are the unmodified SplitData methods, the model is nn_x/w5/model.py Ranker built from the
checkpoint cfg (model.py is byte-identical in nn_x/w5 and nn_x/abl). Batching = common.predict (cand_budget 65536, max_feeds 1024,
no shuffle); numerics = fp32 on CPU (the stored files are bf16 autocast on GPU: compare CPU with CPU only).
xt inputs: every column from nn/feat_cache/train (rows of the holdout feeds) through the attach_xt transform (fp16 storage);
a view replaces the recency-dependent columns by cache/lag_<view>_<name>.npy (raw scale, holdout row order).

Usage: CUDA_VISIBLE_DEVICES="" python infer.py <view> [member,member,...]      view = lag0 | pub | priv | <any built view>
"""
import sys, time
from pl_common import *
import torch
torch.set_num_threads(12)
sys.path.insert(0, str(ROOT / 'nn_x/w5'))
from data import SplitData, load_item_table  # noqa: E402
from model import Ranker  # noqa: E402

MINI = CACHE / 'mini'
T0 = time.time()
lg = lambda *a: print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def build_mini():
    MINI.mkdir(parents=True, exist_ok=True)
    f = feeds('train')
    hold, p = holdout_feeds(f)
    M = f['M'].to_numpy().astype(np.int64)[hold]; H = f['H'].to_numpy().astype(np.int64)[hold]
    co = f['cand_off'].to_numpy().astype(np.int64)[hold]; ho = f['hist_off'].to_numpy().astype(np.int64)[hold]
    meta = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy')
    assert (meta[0] == M).all(), 'holdout identity mismatch'
    crow = rows_of(co, M); hrow = rows_of(ho, H)
    mm = lambda n: np.load(FLAT / 'train' / f'{n}.npy', mmap_mode='r')
    np.save(MINI / 'hold.npy', hold); np.save(MINI / 'crow.npy', crow); np.save(MINI / 'hrow.npy', hrow)
    np.save(MINI / 'M.npy', M); np.save(MINI / 'H.npy', H)
    np.save(MINI / 'user_id.npy', f['user_id'].to_numpy()[hold]); np.save(MINI / 'ts.npy', f['timestamp'].to_numpy().astype(np.int64)[hold])
    np.save(MINI / 'feed_id.npy', f['feed_id'].to_numpy()[hold]); np.save(MINI / 'prand.npy', p[hold])
    np.save(MINI / 'user_emb.npy', np.asarray(mm('user_emb')[hold]))
    ce = np.asarray(mm('cand_emb_idx')[crow]); he = np.asarray(mm('hist_emb_idx')[hrow])
    need = np.unique(np.concatenate([ce[ce >= 0], he[he >= 0]]))
    tab = load_item_table_mm('train')
    np.save(MINI / 'table.npy', np.asarray(tab[need]))
    remap = lambda x: np.where(x >= 0, np.searchsorted(need, np.maximum(x, 0)), -1).astype(np.int32)
    np.save(MINI / 'c_emb.npy', remap(ce)); np.save(MINI / 'h_emb.npy', remap(he))
    np.save(MINI / 'c_cat.npy', np.asarray(mm('cand_category_id')[crow]).astype(np.int16))
    np.save(MINI / 'c_f.npy', np.stack([np.asarray(mm(f'cand_f{i}')[crow]) for i in range(1, 6)], 1))
    np.save(MINI / 'c_item.npy', np.asarray(mm('cand_item_id')[crow]))
    np.save(MINI / 'h_cat.npy', np.asarray(mm('hist_category_id')[hrow]).astype(np.int16))
    np.save(MINI / 'h_et.npy', np.asarray(mm('hist_event_type')[hrow])); np.save(MINI / 'h_es.npy', np.asarray(mm('hist_event_source')[hrow]))
    lg(f'mini split built: {len(hold)} feeds, {len(crow)} cand rows, {len(hrow)} hist rows, {len(need)} table rows')


def load_item_table_mm(split):
    d = ROOT / 'data/competition' / split / 'item_embeddings'
    meta = json.loads((d / 'data.meta').read_text())
    return np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(meta['shape']))


def load_mini():
    if not (MINI / 'h_es.npy').exists():
        build_mini()
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
    s.dom = torch.stack([1 - p, p, torch.zeros_like(p)], 1)  # train.py l.372-373 (train feeds: production soft-mixed by p_random)
    s.crow = L('crow')
    return s


def raw_col(name, crow, view):
    """Raw (untransformed) column on the holdout rows: the view's rebuilt file if it exists, else the stored feat_cache file."""
    p = CACHE / f'lag_{view}_{name}.npy'
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
    s.xt = torch.stack(cols, 1)
    return used


def main():
    view = sys.argv[1]
    mem = sys.argv[2].split(',') if len(sys.argv) > 2 else W8 + POP1KD
    s = load_mini(); lg(f'mini split loaded ({s.n} feeds, {s.n_cand} rows); view {view}')
    cur = None
    for v in mem:
        out_p = CACHE / f'pred_{view}_{v}.npy'
        if out_p.exists():
            lg(f'{v}: exists, skipped'); continue
        ck = torch.load(ckpt_path(v), map_location='cpu', weights_only=False)
        names = ck['cfg']['xt_names']
        if names != cur:
            used = attach(s, names, view); cur = names
            lg(f'xt attached ({len(names)} cols); rebuilt columns used in view {view}: {used}')
            if view != 'lag0':
                exp = [n for n in names if n in LAGCOLS]
                assert sorted(used) == sorted(exp), f'view {view}: rebuilt files missing for {set(exp) - set(used)}'
        m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
        out = torch.zeros(s.n_cand, dtype=torch.float32)
        t1 = time.time()
        with torch.inference_mode():
            for fb in s.batches(np.arange(s.n), 65536, 1024, shuffle=False):
                b = s.gather(fb)
                s.scatter(out, b, m(b, use_pos=False))
        np.save(out_p, out.numpy().astype(np.float32))
        lg(f'{v}: done in {time.time() - t1:.0f}s -> {out_p.name}')


if __name__ == '__main__':
    main()
