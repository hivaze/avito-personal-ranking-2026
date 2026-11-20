# The lagged holdout as a reusable tool. It scores any checkpoint on the 13,424 holdout feeds where the
# gate holds, under three input views (stored, private-like lag, public-like lag), on CPU in fp32 with one
# inference path, and offers paired pool comparisons per view.
# Run as in its docstring: lagged_score.py prep, then lagged_score.py score <views> <members>; the
# read-out is wt_read.py pair '<A>' '<B>' in the lab folder. About 3.2 CPU minutes per member and view.
# Produced cache/pred/<view>/<member>.npy (not copied) and the reads in ../results/witness_*.txt.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/stale/witness/lagged_score.py
#
"""LAGGED HOLDOUT as a tool (task A of KEY = witness).

Scores any nn_x/w5, nn_x/abl, nn_x/r6 or nn/ checkpoint (selected or '@last') on the SUBSET of the in-period holdout
(holdout days >= Sep 13: 13,424 feeds, 2,089,648 rows; the only part where the label-free parity gate of the poplag study holds)
under three input views, on CPU in fp32, with ONE inference path:
   lag0  = the stored inputs (nn/feat_cache/train)                      -> fresh popularity (the in-period view)
   priv  = the study's private-like lagged columns (lag_priv_v3_*)      -> test-like stale popularity, private lags 7/8/9 d
   pub   = the study's public-like lagged columns  (lag_pub_v3_*)       -> public lags 6/7 d
Only the nine recency-dependent columns change between views (seq_cv_{invd20,pop,norm}, it_hist_{all,et1..4}_luo, xp_imp_c50).
Inference = nn_x/w5/data.py SplitData.batches / gather / scatter (unmodified, imported) + nn_x/w5/model.py Ranker from the
checkpoint cfg, domain mix [1 - p_random, p_random, 0] (train feeds), fp32. Batching: the batches of common.predict (65536 / 1024, no shuffle) cut into
anchored sub-batches with identical padding (wt_common.anchored_batches; max |logit diff| 1.5e-5 to the uncut batches, 3.5x faster; WT_REFBATCH=1
restores the uncut batches).
Compare CPU with CPU only (stored GPU bf16 holdout files differ by up to 0.0008 NDCG per member).

  cd <repo> && CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nn_x/w12/stale/witness/lagged_score.py prep
  ... lagged_score.py score <views csv | all> <member,member,... | group name>      # cache/pred/<view>/<member>.npy
  ... lagged_score.py import-staged                                                  # the 12 members staged by poplag / poplag_verify
Read-out functions (pool level per view, paired deltas, D = delta(lagged) - delta(lag 0)): class Lag below; CLI in wt_read.py.
"""
from wt_common import *


# ------------------------------------------------------------------------------------------------------------ subset split
def build_sub():
    """Mini split of the subset feeds from data/flat/train (label-free) + raw xt columns for the three views. Idempotent."""
    if (SUB / 'DONE').exists():
        return
    t0 = time.time()
    f, hold, p = holdout_feeds()
    fM = f['M'].to_numpy().astype(np.int64); fH = f['H'].to_numpy().astype(np.int64)
    fco = f['cand_off'].to_numpy().astype(np.int64); fho = f['hist_off'].to_numpy().astype(np.int64)
    meta = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy')
    assert (meta[0] == fM[hold]).all(), 'holdout identity mismatch'
    hday = (f['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY
    pick = np.flatnonzero(hday >= SEP6 + 7); fx = hold[pick]
    off_h = np.concatenate([[0], np.cumsum(fM[hold])[:-1]])
    hpos = np.concatenate([np.arange(off_h[i], off_h[i] + fM[hold[i]]) for i in pick])       # rows in holdout row order
    crow = np.concatenate([np.arange(fco[k], fco[k] + fM[k]) for k in fx])                   # rows in data/flat/train
    hrow = np.concatenate([np.arange(fho[k], fho[k] + fH[k]) for k in fx])
    assert len(pick) == N_SUB_FEEDS and len(crow) == N_SUB_ROWS, (len(pick), len(crow))
    mm = lambda n: np.load(FLAT / 'train' / f'{n}.npy', mmap_mode='r')
    ce = np.asarray(mm('cand_emb_idx')[crow]); he = np.asarray(mm('hist_emb_idx')[hrow])
    d = ROOT / 'data/competition/train/item_embeddings'; shape = tuple(json.loads((d / 'data.meta').read_text())['shape'])
    tab = np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=shape)
    used = np.unique(np.concatenate([ce[ce >= 0], he[he >= 0]]))
    remap = lambda x: np.where(x >= 0, np.searchsorted(used, np.maximum(x, 0)), -1).astype(np.int32)
    S = lambda n, a: save_atomic(SUB / f'{n}.npy', a)
    S('pick', pick); S('hpos', hpos); S('crow', crow); S('M', fM[fx]); S('H', fH[fx]); S('day', hday[pick])
    S('user_id', f['user_id'].to_numpy()[fx]); S('prand', p[fx].astype(np.float32)); S('user_emb', np.asarray(mm('user_emb')[fx]))
    S('c_emb', remap(ce)); S('h_emb', remap(he)); S('table', np.asarray(tab[used]))
    S('c_cat', np.asarray(mm('cand_category_id')[crow]).astype(np.int16))
    S('c_f', np.stack([np.asarray(mm(f'cand_f{i}')[crow]) for i in range(1, 6)], 1))
    S('h_cat', np.asarray(mm('hist_category_id')[hrow]).astype(np.int16))
    S('h_et', np.asarray(mm('hist_event_type')[hrow])); S('h_es', np.asarray(mm('hist_event_source')[hrow]))
    # lagged columns (raw scale) of the study, cut to the subset rows; self-contained copies
    for view, tag in VIEWS.items():
        if tag is None:
            continue
        for c in LAGCOLS:
            x = np.load(STUDY / f'lag_{tag}_{c}.npy', mmap_mode='r'); assert len(x) == int(meta[0].sum()), (c, len(x))
            S(f'xt_{view}_{c}', np.asarray(x[hpos]))
    # imposed lags and the cold share (lagged it_hist_all_luo == 0) per subset feed, for slices
    for view, tag in (('priv', 'priv'), ('pub', 'pub')):
        S(f'L_{view}', np.load(STUDY / f'view_{tag}_L.npy')[pick])
    (SUB / 'DONE').write_text(f'{now()} subset mini split: {len(pick)} feeds, {len(crow)} rows, {len(hrow)} history events, {len(used)} table rows; '
                              f'{time.time() - t0:.0f}s\n')
    print((SUB / 'DONE').read_text(), flush=True)


def raw_col(name, view):
    """Raw column on the subset rows: the view's lagged copy for the nine recency-dependent columns, else the stored feat_cache column (cached)."""
    if view != 'lag0' and name in LAGCOLS:
        return np.load(SUB / f'xt_{view}_{name}.npy'), True
    p = SUB / f'xt_lag0_{name}.npy'
    if not p.exists():
        crow = np.load(SUB / 'crow.npy')
        save_atomic(p, np.asarray(np.load(FC / 'train' / f'{name}.npy', mmap_mode='r')[crow]))
    return np.load(p), False


class SubSplit:
    """The subset feeds in the SplitData attribute layout (CPU). attach(names, view) sets .xt; score(model) returns flat fp32 logits."""

    def __init__(self):
        import torch
        sys.path.insert(0, str(ROOT / 'nn_x/w5'))
        from data import SplitData
        build_sub()
        L = lambda n: np.load(SUB / f'{n}.npy')
        g = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        s = SplitData.__new__(SplitData)
        s.split = 'witness_subset'; s.device = torch.device('cpu'); s.max_hist = 256
        s.M_np = L('M'); s.H_np = L('H'); s.n = len(s.M_np)
        s.cand_off_np = np.concatenate([[0], np.cumsum(s.M_np)[:-1]]).astype(np.int64)
        s.user_id = L('user_id')
        s.M = g(s.M_np); s.H = g(s.H_np); s.cand_off = g(s.cand_off_np)
        s.hist_off = g(np.concatenate([[0], np.cumsum(s.H_np)[:-1]]).astype(np.int64))
        s.user_emb = g(L('user_emb')); s.user0 = g(s.user_id == 0)
        s.c_emb = g(L('c_emb')); s.h_emb = g(L('h_emb')); s.table = g(L('table'))
        s.n_cand = len(s.c_emb); s.n_hist = len(s.h_emb)
        s.c_cat = g(L('c_cat')); s.c_f = g(L('c_f')); s.h_cat = g(L('h_cat')); s.h_et = g(L('h_et')); s.h_es = g(L('h_es'))
        s.c_item = None; s.h_item = None; s.xt = None; s.has_target = False; s.target_np = None; s.c_tgt = None; s.has_pos = False; s.c_pos = None
        p = g(L('prand').astype(np.float32))
        s.dom = torch.stack([1 - p, p, torch.zeros_like(p)], 1)              # nn_x/w5/train.py l.372-373 (train feeds)
        self.s = s; self.key = None; self.torch = torch
        self.bl = None if os.environ.get('WT_REFBATCH') else anchored_batches(s)      # same padding as the reference batches, 3.5x faster on CPU

    def attach(self, names, view):
        key = (tuple(names), view)
        if key == self.key:
            return self.used
        cols, used = [], []
        for n in names:
            base = n[4:] if n.startswith('has:') else n
            x, u = raw_col(base, view)
            x = (~np.isnan(x)).astype(np.float32) if n.startswith('has:') else xt_transform(n, x)
            if u: used.append(n)
            cols.append(self.torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(self.torch.float16))
        self.s.xt = self.torch.stack(cols, 1); self.key = key; self.used = used
        if view != 'lag0':
            exp = [n for n in names if n in LAGCOLS]
            assert sorted(used) == sorted(exp), f'view {view}: lagged columns not applied for {set(exp) - set(used)}'
        return used

    def score(self, m):
        torch = self.torch; s = self.s
        out = torch.zeros(s.n_cand, dtype=torch.float32)
        with torch.inference_mode():
            for fb in (self.bl if self.bl is not None else s.batches(np.arange(s.n), 65536, 1024, shuffle=False)):
                b = s.gather(fb)
                s.scatter(out, b, m(b, use_pos=False))
        return out.numpy().astype(np.float32)


def load_model(v):
    import torch
    sys.path.insert(0, str(ROOT / 'nn_x/w5'))
    from model import Ranker
    ck = torch.load(ckpt_path(v), map_location='cpu', weights_only=False)
    assert not ck['cfg'].get('id_buckets', 0), f'{v}: item-id models are not supported by the mini splits'
    m = Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval()
    return m, ck['cfg']['xt_names']


def pred_path(view, v):
    return PRED / view / f'{v}.npy'


def score_member(sub, v, views, log=print):
    """Score one member on the requested views (skips cached ones). Returns seconds per view actually scored."""
    todo = [w for w in views if not pred_path(w, v).exists()]
    if not todo:
        return {}
    m, names = load_model(v)
    sec = {}
    for w in todo:
        t1 = time.time()
        used = sub.attach(names, w)
        save_atomic(pred_path(w, v), sub.score(m))
        sec[w] = time.time() - t1
        log(f'{now()} lagged {v} view {w}: {sec[w]:.0f}s ({len(used)} lagged columns applied)')
    return sec


def import_staged(log=print):
    """Cut the predictions staged by the poplag study (lag0, priv; 30,000 feeds) and by the reviewer (pub; subset) to my layout.
    Same inference path (CPU fp32, SplitData + Ranker); written to cache/pred/<view>/<member>.npy and listed in cache/pred/IMPORTED.txt."""
    build_sub()
    hpos = np.load(SUB / 'hpos.npy'); lines = []
    for v in G['W8'] + G['pop1kd']:
        for view, src in (('lag0', STUDY / f'pred_lag0_{v}.npy'), ('priv', STUDY / f'pred_priv_v3_{v}.npy'), ('pub', VERIFY / f'pred_pub_v3_late_{v}.npy')):
            out = pred_path(view, v)
            if out.exists() or not src.exists():
                continue
            x = np.load(src); x = x[hpos] if len(x) != N_SUB_ROWS else x
            assert len(x) == N_SUB_ROWS
            save_atomic(out, x.astype(np.float32)); lines.append(f'{now()} {view} {v} <- {src}')
    if lines:
        with open(PRED / 'IMPORTED.txt', 'a') as fh:
            fh.write('\n'.join(lines) + '\n')
    log(f'imported {len(lines)} staged prediction files')


# ------------------------------------------------------------------------------------------------------------------ read-out
class Lag:
    """Read-out on the subset. Pools are specs: a list of member names, or a list of (weight, [members]) groups
    (weighted pool = sum_g w_g * z(mean of member z)).  level(spec, view) -> per-feed NDCG;  compare(A, B) -> dict of (mean, SE)."""

    def __init__(self):
        build_sub()
        self.M = np.load(SUB / 'M.npy').astype(np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.M)[:-1]]).astype(np.int64)
        self.hpos = np.load(SUB / 'hpos.npy'); self.day = np.load(SUB / 'day.npy')
        self.y = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)[self.hpos]
        self._z = {}

    def have(self, view, v):
        return pred_path(view, v).exists() if view != 'stored' else stored_holdout(v).exists()

    def z(self, view, v):
        k = (view, v)
        if k not in self._z:
            x = np.load(stored_holdout(v))[self.hpos] if view == 'stored' else np.load(pred_path(view, v))
            self._z[k] = feed_z(self.off, self.M, x)
        return self._z[k]

    def pool(self, spec, view):
        if isinstance(spec[0], str):
            return np.mean([self.z(view, v) for v in spec], 0)
        acc = 0.0
        for g in spec:      # (w, members) -> w * z(mean member z);  (w, members, 'raw') -> w * mean member z (no re-standardisation)
            mz = np.mean([self.z(view, v) for v in g[1]], 0)
            acc = acc + g[0] * (mz if len(g) > 2 and g[2] == 'raw' else feed_z(self.off, self.M, mz)).astype(np.float64)
        return acc

    def level(self, spec, view):
        return ndcg_fast(self.pool(spec, view), self.y, self.off, self.M)

    def member_level(self, v, view):
        return ndcg_fast(self.z(view, v), self.y, self.off, self.M)

    def compare(self, A, B, views=('lag0', 'priv', 'pub')):
        """Paired deltas A - B per view, D_<view> = delta(view) - delta(lag0), and pool level changes under the lag."""
        r = {}; lv = {}
        for w in views:
            a, b = self.level(A, w), self.level(B, w); lv[w] = (a, b)
            r[f'A_{w}'] = float(a.mean()); r[f'B_{w}'] = float(b.mean()); r[f'd_{w}'] = ms(a - b)
        for w in views:
            if w != 'lag0' and 'lag0' in lv:
                r[f'D_{w}'] = ms((lv[w][0] - lv[w][1]) - (lv['lag0'][0] - lv['lag0'][1]))
                r[f'dropA_{w}'] = ms(lv[w][0] - lv['lag0'][0]); r[f'dropB_{w}'] = ms(lv[w][1] - lv['lag0'][1])
        r['n'] = len(self.M); r['_per_feed'] = {w: lv[w][0] - lv[w][1] for w in views}
        return r


def main():
    cmd = sys.argv[1]
    if cmd == 'prep':
        build_sub(); return
    if cmd == 'import-staged':
        import_staged(); return
    assert cmd == 'score', cmd
    import torch
    torch.set_num_threads(int(NT))
    views = list(VIEWS) if sys.argv[2] == 'all' else sys.argv[2].split(',')
    mem = []
    for tok in sys.argv[3].split(','):
        mem += G[tok] if tok in G else [tok]
    sub = SubSplit()
    for v in mem:
        if not ckpt_path(v).exists():
            print(f'{v}: no checkpoint, skipped', flush=True); continue
        score_member(sub, v, views)


if __name__ == '__main__':
    main()
