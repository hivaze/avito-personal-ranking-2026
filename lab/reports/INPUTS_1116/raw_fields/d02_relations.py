"""INVENTORY D step 1b: relations between raw columns that could carry information (CPU only, read-only).
Output: reports/INPUTS_1116/raw_fields/d02_relations.txt"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
import numpy as np, polars as pl, json
from pathlib import Path
from scipy.stats import spearmanr, rankdata
R = Path('<repo>'); F = R / 'data/flat'; OUT = R / 'reports/INPUTS_1116/raw_fields'
L = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); L.append(s)
ld = lambda sp, n: np.load(F / sp / f'{n}.npy', mmap_mode='r')
knots = np.load(R / 'data/feats/life_f4_age_knots.npy'); K = 64; CAP = 63 / 64 - 1e-6
def age_hours(x):
    x = x.astype(np.float64); u = x * K; k = np.clip(np.floor(u).astype(int), 1, K - 2); w = u - k
    return np.where(x < CAP, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)
fe = {sp: pl.read_parquet(F / sp / 'feeds.parquet') for sp in ('train', 'val', 'test')}
# ---------- A: item_id vs posting time; B: emb_idx vs item_id
for sp in ('val', 'test'):
    f = fe[sp]; fi = np.asarray(ld(sp, 'cand_fidx')); it = np.asarray(ld(sp, 'cand_item_id')); f4 = np.asarray(ld(sp, 'cand_f4'))
    t = f['timestamp'].to_numpy()[fi]; age = age_hours(f4); ok = np.isfinite(age); tp = t - age * 3600
    P(f'[A] {sp}: Spearman(item_id, posting time) over candidates with decodable age ({ok.mean():.4f}): {spearmanr(it[ok], tp[ok])[0]:+.4f} | Spearman(item_id, f4) {spearmanr(it, f4)[0]:+.4f}')
    ei = np.asarray(ld(sp, 'cand_emb_idx')); m = ei >= 0
    P(f'[B] {sp}: Spearman(emb_idx, item_id) cand {spearmanr(ei[m], it[m])[0]:+.4f} | Spearman(emb_idx, row order of first use) {spearmanr(ei[m], np.arange(len(ei))[m])[0]:+.4f} | Spearman(emb_idx, posting time) {spearmanr(ei[m & ok], tp[m & ok])[0]:+.4f}')
    hi = np.asarray(ld(sp, 'hist_item_id')); he = np.asarray(ld(sp, 'hist_emb_idx')); mh = he >= 0
    P(f'[B] {sp}: Spearman(emb_idx, item_id) hist {spearmanr(he[mh][::7], hi[mh][::7])[0]:+.4f}')
    # item_id -> one emb_idx? and emb_idx -> one item?
    d = pl.DataFrame({'it': np.r_[it[m], hi[mh]], 'e': np.r_[ei[m], he[mh]]}).unique()
    P(f'[B] {sp}: distinct (item, emb_idx) pairs {d.height} | distinct items {d["it"].n_unique()} | distinct emb_idx {d["e"].n_unique()}')
    # D: duplicates of item within feed
    key = fi.astype(np.int64) * (1 << 31) + it
    P(f'[D] {sp}: candidate rows whose (feed, item) is duplicated: {1 - len(np.unique(key)) / len(key):.6f}')
    # E: within-feed tie groups of f1..f5
    for c in ('f1', 'f2', 'f3', 'f4', 'f5'):
        x = np.asarray(ld(sp, f'cand_{c}')); d = pl.DataFrame({'f': fi, 'x': x, 'c': np.asarray(ld(sp, 'cand_category_id'))})
        g = d.group_by(['f', 'x']).agg(pl.len().alias('n'), pl.col('c').n_unique().alias('nc'))
        n = g['n'].to_numpy()
        P(f'[E] {sp} {c}: distinct (feed, value) groups {g.height} for {len(x)} rows | rows in a tie group of size>=2: {n[n >= 2].sum() / len(x):.4f} | size>=5: {n[n >= 5].sum() / len(x):.4f} | groups(size>=2) spanning >1 category: {np.mean(g.filter(pl.col("n") >= 2)["nc"].to_numpy() > 1):.4f}')
    # C: user_id vs things
    u = f['user_id'].to_numpy(); nz = u > 0
    P(f'[C] {sp}: Spearman(user_id, H) {spearmanr(u[nz], f["H"].to_numpy()[nz])[0]:+.4f} | Spearman(user_id, M) {spearmanr(u[nz], f["M"].to_numpy()[nz])[0]:+.4f} | Spearman(user_id, timestamp) {spearmanr(u[nz], f["timestamp"].to_numpy()[nz])[0]:+.4f} | Spearman(feed_id, M) {spearmanr(f["feed_id"].to_numpy(), f["M"].to_numpy())[0]:+.4f} Spearman(feed_id, user_id) {spearmanr(f["feed_id"].to_numpy()[nz], u[nz])[0]:+.4f}')
# H: val row index vs label
f = fe['val']; fi = np.asarray(ld('val', 'cand_fidx')); y = np.asarray(ld('val', 'cand_target')); M = f['M'].to_numpy(); off = f['cand_off'].to_numpy()
idx = np.arange(len(fi)) - off[fi]; rel = idx / np.maximum(M[fi] - 1, 1)
P(f'[H] val: mean relative row index of positives {rel[y == 1].mean():.4f} (0.5 = none); positive rate at index 0 {y[idx == 0].mean():.4f}, at last index {y[idx == M[fi] - 1].mean():.4f}, overall {y.mean():.4f} | mean feed positive rate vs feed_id Spearman {spearmanr(f["feed_id"].to_numpy(), np.add.reduceat(y.astype(np.float64), off) / M)[0]:+.4f}')
# item_id of val: posting time monotone? binned view
it = np.asarray(ld('val', 'cand_item_id')); f4 = np.asarray(ld('val', 'cand_f4')); t = f['timestamp'].to_numpy()[fi]; age = age_hours(f4); ok = np.isfinite(age)
q = np.quantile(it[ok], np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, it[ok]) - 1, 0, 9)
P('[A] val: median listing age (h) by item_id decile: ' + ' '.join(f'{np.median(age[ok][b == k]):.0f}' for k in range(10)))
u = f['user_id'].to_numpy()
# user ids: train membership vs id
ut = np.unique(fe['train']['user_id'].to_numpy())
for sp in ('val', 'test'):
    uu = np.unique(fe[sp]['user_id'].to_numpy()); uu = uu[uu > 0]; intr = np.isin(uu, ut)
    P(f'[C] {sp}: non-zero users {len(uu)} | in train {intr.mean():.4f} | median user_id in-train {np.median(uu[intr]):.0f} vs not-in-train {np.median(uu[~intr]):.0f} | max train user_id {ut.max()}')
tf = fe['train']; tu = tf['user_id'].to_numpy(); tt = tf['timestamp'].to_numpy()
first = pl.DataFrame({'u': tu, 't': tt}).filter(pl.col('u') > 0).group_by('u').agg(pl.col('t').min())
P(f'[C] train: Spearman(user_id, first feed time) {spearmanr(first["u"].to_numpy(), first["t"].to_numpy())[0]:+.4f} | Spearman(user_id, H) {spearmanr(tu[tu > 0][::20], tf["H"].to_numpy()[tu > 0][::20])[0]:+.4f}')
# ---------- F: missing embeddings recoverable from other splits' tables
tr_ci = np.asarray(ld('train', 'cand_item_id')); tr_ce = np.asarray(ld('train', 'cand_emb_idx'))
tr_has = np.unique(tr_ci[tr_ce >= 0]); P(f'[F] train candidate items with an embedding: {len(tr_has)}')
tr_hi = ld('train', 'hist_item_id'); tr_he = ld('train', 'hist_emb_idx')
NH = len(tr_hi); miss_tr = 0; parts = []
for a in range(0, NH, 100_000_000):
    hi_ = np.asarray(tr_hi[a:a + 100_000_000]); he_ = np.asarray(tr_he[a:a + 100_000_000]); miss_tr += int((he_ < 0).sum()); parts.append(np.unique(hi_[he_ >= 0]))
tr_hhas = np.unique(np.concatenate(parts)); del parts
P(f'[F] train history events without embedding: {miss_tr / NH:.4f} | train history items with an embedding: {len(tr_hhas)}')
tr_any = np.union1d(tr_has, tr_hhas)
for sp in ('val', 'test'):
    hi = np.asarray(ld(sp, 'hist_item_id')); he = np.asarray(ld(sp, 'hist_emb_idx')); ci = np.asarray(ld(sp, 'cand_item_id')); ce = np.asarray(ld(sp, 'cand_emb_idx'))
    mm = he < 0
    other = 'test' if sp == 'val' else 'val'
    oi = np.r_[np.asarray(ld(other, 'cand_item_id'))[np.asarray(ld(other, 'cand_emb_idx')) >= 0], np.asarray(ld(other, 'hist_item_id'))[np.asarray(ld(other, 'hist_emb_idx')) >= 0]]
    own = np.unique(np.r_[ci[ce >= 0], hi[he >= 0]])
    P(f'[F] {sp}: history events without embedding {mm.mean():.4f} | of those: item has an embedding in the train table {np.isin(hi[mm], tr_any).mean():.4f}, in the {other} table {np.isin(hi[mm], oi).mean():.4f}, elsewhere in the own table {np.isin(hi[mm], own).mean():.4f} | missing-emb events by hpos<32: {mm[np.asarray(ld(sp, "hist_hpos")) < 32].mean():.4f}')
    et = np.asarray(ld(sp, 'hist_event_type'))
    P(f'[F] {sp}: missing-embedding share by event type: ' + ' '.join(f'e{e}:{mm[et == e].mean():.3f}' for e in range(1, 11)))
    P(f'[F] {sp}: candidates without embedding {np.mean(ce < 0):.5f}; recoverable from train table {np.isin(ci[ce < 0], tr_any).mean():.4f}')
    # J: history items that are candidates somewhere
    P(f'[J] {sp}: history events whose item is a train candidate (with or without emb): {np.isin(hi, np.unique(tr_ci)).mean():.4f}')
del tr_any, tr_hhas, tr_has
# ---------- G: same-user earlier feeds (train) vs val/test candidates
tfi = ld('train', 'cand_fidx')
for sp in ('val', 'test'):
    f = fe[sp]; u = f['user_id'].to_numpy(); uu = np.unique(u[u > 0])
    tfeeds = np.flatnonzero(np.isin(tu, uu)); P(f'[G] {sp}: train feeds of {sp} users: {len(tfeeds)} (users {len(np.unique(tu[tfeeds]))})')
    toff = tf['cand_off'].to_numpy(); tM = tf['M'].to_numpy()
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(toff[tfeeds], tM[tfeeds])])
    tr = pl.DataFrame({'u': tu[np.asarray(tfi[rows])], 'it': tr_ci[rows], 'ty': np.asarray(ld('train', 'cand_target')[rows]), 'tpos': np.asarray(ld('train', 'cand_pos')[rows]), 'tt': tt[np.asarray(tfi[rows])]})
    fi = np.asarray(ld(sp, 'cand_fidx')); d = pl.DataFrame({'row': np.arange(len(fi)), 'u': u[fi], 'it': np.asarray(ld(sp, 'cand_item_id'))})
    if sp == 'val': d = d.with_columns(pl.Series('y', np.asarray(ld('val', 'cand_target'))))
    j = d.join(tr, on=['u', 'it'], how='inner')
    seen_feeds = np.isin(u, np.unique(tu[tfeeds]))
    P(f'[G] {sp}: feeds whose user has a train feed {seen_feeds.mean():.4f} | candidate rows also shown to the same user in a train feed: {j["row"].n_unique()} = {j["row"].n_unique() / len(fi):.5f} of rows, {j["row"].n_unique() / max(1, int(seen_feeds[fi].sum())):.5f} of rows of linked feeds')
    if sp == 'val' and j.height:
        jj = j.unique('row'); yv = np.asarray(ld('val', 'cand_target'))
        P(f'[G] val: positive rate of re-shown rows {jj["y"].mean():.4f} (n {jj.height}; earlier label positive share {jj["ty"].mean():.4f}) vs all rows of linked feeds {yv[seen_feeds[fi]].mean():.4f}')
    # history-side: val/test candidate in an EARLIER train history snapshot of the same user is impossible to differ from own history (snapshots nest) -> skipped
(OUT / 'd02_relations.txt').write_text('\n'.join(L) + '\n')
