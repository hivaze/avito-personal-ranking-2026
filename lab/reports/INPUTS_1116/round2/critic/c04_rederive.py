"""CRITIC2 c04: independent re-derivation of the rescue arrays on sampled rows (my own polars / numpy path, no kernel of the explorer), and a
label-leak test for the label statistic.
 A  kn1_5: for 6 random content clusters, the index of every window is rebuilt from the raw train candidate log (non-holdout feeds of the window's
    slots, >= 5 impressions, <= 1500 most shown per cluster, ties by item id), and the highest cosine (own item skipped) is recomputed for sampled
    rows of the five query sets.  Also: the largest slot that contributes to a row's index against the row's own slot.
 B  ns2_5: for sampled rows, the number of distinct OTHER users with a type-1, source-2 history event on the item in a snapshot of a train feed of
    the window, from the raw history arrays.
 C  bt5 (label statistic): (i) seen flag from the raw arrays: item is a candidate of a source feed (p_random > 0.5, >= 1 positive, not holdout) of
    the window;  (ii) own-feed leak test: if a row's own labels entered its value, the column would predict the click far better on POOL rows (their
    feeds ARE sources of other rows' windows) than on HOLDOUT rows (never a source).  Mean within-feed AUC of bt5 and coec5 for the click, pool against
    holdout, by slot group.  (labels are opened in part C only)
usage: c04_rederive.py    output out/c04_rederive.txt"""
import json
from cc_common import *
P = Log(OUT / 'c04_rederive.txt')
V = rs.Views(P)
a = V.A['train']; rng = np.random.default_rng(31337)
CLU = ROOT / 'data/clu_work'
QS = ('hold', 'pool', 'val_nat', 'val_twin', 'test_nat')
R = {q: V.rows(q) for q in QS}
CUTROW = {q: np.repeat(V.Q[q][2], R[q][3]) for q in QS}
SLOTROW = {q: np.repeat(V.A[V.Q[q][0]]['slot'][V.Q[q][1]], R[q][3]) for q in QS}
dense_feed = rs.S12[a['slot']] >= 0


def emb_table(split):
    d = ROOT / f'data/competition/{split}/item_embeddings'; mt = json.loads((d / 'data.meta').read_text())
    return np.memmap(d / 'data.memmap', dtype=np.int8, mode='r', shape=tuple(mt['shape']))


# ------------------------------------------------------------------ A
P('=== A. kn1_5 from the raw candidate log ===')
cl_tr = np.load(CLU / 'cand_cl_4096_train.npy'); pick = np.sort(rng.choice(4096, 6, replace=False)); lut = np.zeros(4097, bool); lut[pick + 1] = True
m = lut[cl_tr.astype(np.int64) + 1]; rr = np.nonzero(m)[0]
P(f'clusters {pick.tolist()}: train candidate rows {len(rr)}')
it = np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r')[rr]; fx = np.load(FLAT / 'train/cand_fidx.npy', mmap_mode='r')[rr]; ei = np.load(FLAT / 'train/cand_emb_idx.npy', mmap_mode='r')[rr]
T = pl.DataFrame({'item': np.asarray(it).astype(np.int64), 'cl': cl_tr[rr].astype(np.int64), 'slot': a['slot'][fx], 'hold': V.ish[fx], 'emb': np.asarray(ei).astype(np.int64)})
meta = T.group_by('item').agg(pl.col('cl').n_unique().alias('ncl'), pl.col('emb').n_unique().alias('nemb'), pl.col('cl').last(), pl.col('emb').last())
P(f'items {len(meta)}; items with more than one cluster / more than one embedding row among their train rows: {int((meta["ncl"] > 1).sum())} / {int((meta["nemb"] > 1).sum())}')
cnt = T.filter(~pl.col('hold') & pl.col('slot').is_in(rs.DENSE.tolist())).group_by(['item', 'slot']).len()
ET = {s: emb_table(s) for s in ('train', 'val', 'test')}
nbad = 0; ntot = 0; worst = 0.0
for q in QS:
    sp = V.Q[q][0]; rows = R[q][0]
    clq = np.asarray(np.load(CLU / f'cand_cl_4096_{sp}.npy', mmap_mode='r')[rows]).astype(np.int64)
    cand = np.nonzero(np.isin(clq, pick) & (CUTROW[q] >= 0))[0]; cand = np.sort(rng.choice(cand, min(150, len(cand)), replace=False))
    itq = np.asarray(np.load(FLAT / sp / 'cand_item_id.npy', mmap_mode='r')[rows[cand]]).astype(np.int64); eq = np.asarray(np.load(FLAT / sp / 'cand_emb_idx.npy', mmap_mode='r')[rows[cand]]).astype(np.int64)
    stored = rget(q, 'kn1_5')[cand]; mine = np.full(len(cand), np.nan); newest = np.full(len(cand), -1)
    for cut in np.unique(CUTROW[q][cand]):
        W = rs.window(int(cut)).tolist()
        cw = cnt.filter(pl.col('slot').is_in(W)).group_by('item').agg(pl.col('len').sum().alias('cw'), pl.col('slot').max().alias('smax')).join(meta, on='item')
        cw = cw.filter((pl.col('cw') >= 5) & (pl.col('emb') >= 0) & (pl.col('cl') >= 0)).sort(['cl', 'cw', 'item'], descending=[False, True, False])
        cw = cw.with_columns(pl.col('item').cum_count().over('cl').alias('rk')).filter(pl.col('rk') <= 1500)
        for j in np.nonzero(CUTROW[q][cand] == cut)[0]:
            if eq[j] < 0:
                continue
            ix = cw.filter((pl.col('cl') == clq[cand[j]]) & (pl.col('item') != itq[j]))
            if len(ix) == 0:
                continue
            E = np.asarray(ET['train'][np.sort(ix['emb'].to_numpy())], np.float64); v = np.asarray(ET[sp][eq[j]], np.float64)
            mine[j] = float((E @ v / (np.sqrt((E ** 2).sum(1)) * np.sqrt((v ** 2).sum()))).max()); newest[j] = int(ix['smax'].max())
    d = np.abs(mine - stored); both = ~np.isnan(mine) & ~np.isnan(stored); nanm = int((np.isnan(mine) != np.isnan(stored)).sum())
    bad = int((d[both] > 2e-5).sum()) + nanm; nbad += bad; ntot += len(cand); worst = max(worst, float(d[both].max()) if both.any() else 0.0)
    P(f'  {q}: rows {len(cand)}, cuts {np.unique(CUTROW[q][cand]).tolist()}; max |mine - stored| {d[both].max():.2e}; NaN-pattern mismatches {nanm}; rows differing by more than 2e-5: {bad}; '
      f'min (row slot - newest slot in its index) {int((SLOTROW[q][cand] - newest)[newest >= 0].min())} d')
P(f'A: {ntot} rows, {nbad} mismatches, largest difference {worst:.2e}')
del cl_tr, T, cnt

# ------------------------------------------------------------------ B
P('=== B. ns2_5 from the raw history arrays ===')
S = {}
for q in QS:
    sp = V.Q[q][0]; rows = R[q][0]; x = rget(q, 'ns2_5'); ok = np.nonzero(CUTROW[q] >= 0)[0]
    pos = ok[x[ok] > 0]; c = np.concatenate([rng.choice(pos, 120, replace=False), rng.choice(ok, 80, replace=False)]); c = np.unique(c)
    S[q] = dict(c=c, item=np.asarray(np.load(FLAT / sp / 'cand_item_id.npy', mmap_mode='r')[rows[c]]).astype(np.int64), u=V.A[sp]['u'][V.Q[q][1]][R[q][1][c]], cut=CUTROW[q][c], stored=x[c], slot=SLOTROW[q][c])
want = np.unique(np.concatenate([S[q]['item'] for q in QS]))
hi = np.load(FLAT / 'train/hist_item_id.npy'); lutb = np.zeros(int(max(hi.max(), want.max())) + 1, bool); lutb[want] = True
mk = lutb[hi]; idx = np.nonzero(mk)[0]; del mk
t = np.load(FLAT / 'train/hist_event_type.npy', mmap_mode='r')[idx]; so = np.load(FLAT / 'train/hist_event_source.npy', mmap_mode='r')[idx]; hf = np.asarray(np.load(FLAT / 'train/hist_fidx.npy', mmap_mode='r')[idx]).astype(np.int64)
k2 = (np.asarray(t) == 1) & (np.asarray(so) == 2)
H = pl.DataFrame({'item': hi[idx][k2].astype(np.int64), 'f': hf[k2]}); del hi
uu = a['u'][H['f'].to_numpy()]; H = H.with_columns(pl.Series('uk', np.where(uu > 0, uu, -(H['f'].to_numpy() + 1))), pl.Series('slot', a['slot'][H['f'].to_numpy()]))
P(f'history events on the {len(want)} sampled items: {len(idx)}; type 1 and source 2: {len(H)}')
nbad = 0; ntot = 0
for q in QS:
    s = S[q]; mine = np.zeros(len(s['c'])); newest = np.full(len(s['c']), -1)
    for j in range(len(s['c'])):
        W = rs.window(int(s['cut'][j])).tolist()
        g = H.filter((pl.col('item') == int(s['item'][j])) & pl.col('slot').is_in(W) & (pl.col('uk') != int(s['u'][j]) if s['u'][j] > 0 else pl.lit(True)))
        mine[j] = g['uk'].n_unique(); newest[j] = int(g['slot'].max()) if len(g) else -1
    bad = int((mine != s['stored']).sum()); nbad += bad; ntot += len(mine)
    P(f'  {q}: rows {len(mine)} (stored > 0 on {int((s["stored"] > 0).sum())}); mismatches {bad}; min (row slot - newest contributing slot) {int((s["slot"] - newest)[newest >= 0].min())} d')
P(f'B: {ntot} rows, {nbad} mismatches')
del H

# ------------------------------------------------------------------ C
P('=== C. bt5: seen flag from the raw arrays, and the own-feed leak test (labels opened here) ===')
praw = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0)
y = np.load(FLAT / 'train/cand_target.npy'); npos_f = np.add.reduceat(y.astype(np.int64), a['co'])
srcf = (praw > 0.5) & ~V.ish & (npos_f >= 1) & dense_feed
P(f'source feeds (p_random > 0.5, not holdout, >= 1 positive, dense slot): {int(srcf.sum())}')
item = np.load(FLAT / 'train/cand_item_id.npy'); fidx = np.load(FLAT / 'train/cand_fidx.npy')
sr = np.nonzero(srcf[fidx])[0]; SI = pl.DataFrame({'item': item[sr].astype(np.int64), 'slot': a['slot'][fidx[sr]]}).unique()
del sr
nbad = 0; ntot = 0
for q in QS:
    sp = V.Q[q][0]; rows = R[q][0]; ok = np.nonzero(CUTROW[q] >= 0)[0]; c = np.sort(rng.choice(ok, 400, replace=False))
    itq = np.asarray(np.load(FLAT / sp / 'cand_item_id.npy', mmap_mode='r')[rows[c]]).astype(np.int64); stored = rget(q, 'bt5_seen')[c]; th = rget(q, 'bt5')[c]
    g = SI.filter(pl.col('item').is_in(np.unique(itq).tolist()))
    mine = np.zeros(len(c))
    for j in range(len(c)):
        W = rs.window(int(CUTROW[q][c[j]])).tolist(); mine[j] = len(g.filter((pl.col('item') == int(itq[j])) & pl.col('slot').is_in(W))) > 0
    bad = int((mine != stored).sum()); nbad += bad; ntot += len(c)
    P(f'  {q}: rows {len(c)}, seen {stored.mean():.3f}; seen-flag mismatches {bad}; rows with theta != 0 while unseen {int(((th != 0) & (stored == 0)).sum())}')
P(f'C(i): {ntot} rows, {nbad} mismatches')
del item, fidx, SI


@nb.njit(parallel=True, cache=True)
def feed_auc(x, y, off, M):
    out = np.full(len(M), np.nan)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; npos = 0
        for i in range(m):
            if y[o + i] > 0: npos += 1
        if npos == 0 or npos == m: continue
        s = 0.0
        for i in range(m):
            if y[o + i] > 0:
                for j in range(m):
                    if y[o + j] == 0:
                        if x[o + i] > x[o + j]: s += 1.0
                        elif x[o + i] == x[o + j]: s += 0.5
        out[f] = s / (npos * (m - npos))
    return out


P('C(ii) own-feed leak test: mean within-feed AUC of the column for the click (feeds with a view and >= 1 positive), +- 1 SE over feeds')
for col in ('bt5', 'coec5', 'ctr5', 'kn1_5', 'ns2_5'):
    cells = []
    for q in ('pool', 'hold'):
        rows, fl, off, M = R[q]; yq = np.asarray(y[rows]).astype(np.float64); x = np.nan_to_num(rget(q, col), nan=0.0); sl = V.A['train']['slot'][V.Q[q][1]]; hasv = V.Q[q][2] >= 0
        au = feed_auc(x, yq, off, M)
        for nm_, mk in (('slots 6-8', (sl >= 6) & (sl <= 8) & hasv), ('slots 11-13', (sl >= 11) & hasv)):
            v = au[mk & ~np.isnan(au)]; cells.append(f'{q} {nm_}: {v.mean():.4f} +- {v.std() / np.sqrt(len(v)):.4f} ({len(v)} feeds)')
    P(f'  {col:6s} | ' + ' | '.join(cells))
P('DONE')
