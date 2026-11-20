"""poplag_verify step B: lags, my own rebuild of xp_imp_c50 under the declared v3 construction, and the gate numbers. LABEL-FREE (no target file opened).
The construction is re-implemented from the text of nn_x/w12/next/poplag/DECLARATION.txt, not from the study's code.
  cd <repo>/nn_x/w12/next/poplag_verify && CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_gate.py"""
from v_common import *
import time, datetime, json

T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:5.0f}s]', *a, flush=True)


assert datetime.date(1970, 1, 1) + datetime.timedelta(days=SEP6) == datetime.date(2026, 9, 6)
ND = 15                                              # day columns Sep 6 .. Sep 20
EDGES = np.array([0, 1, 3, 7, 10, 24, 76, np.inf]); BN = ['<1d', '1-3d', '3-7d', '7-10d', '10-24d', '24-76d', '>76d+cap']
KN = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')


def age_h(f4):
    """f4 -> listing age in hours: linear between the 64 knots at f4 = k/64 (arithmetic of feat_new/hz_common.py age_hours, re-typed); NaN at the cap 63/64."""
    u = f4.astype(np.float64) * 64.0
    k = np.clip(np.floor(u).astype(np.int64), 1, 62)
    a = KN[k] + (u - k) * (KN[k + 1] - KN[k])
    return np.where(f4.astype(np.float64) < 63 / 64, a, np.nan)


def bucket(ah):
    d = np.where(np.isnan(ah), 1e9, ah / 24.0)
    return np.clip(np.searchsorted(EDGES, d, side='right') - 1, 0, 6)


# ------------------------------------------------------------------ 1. facts
ftr, hold, _ = my_holdout()
fts = ftr['timestamp'].to_numpy().astype(np.int64); fday = (fts + TZ) // DAY; fuser = ftr['user_id'].to_numpy().astype(np.int64)
fM = ftr['M'].to_numpy().astype(np.int64); fco = ftr['cand_off'].to_numpy().astype(np.int64); ffid = ftr['feed_id'].to_numpy()
u, c = np.unique(fday, return_counts=True)
lg('train feeds per MSK day:', {str(datetime.date(1970, 1, 1) + datetime.timedelta(days=int(a))): int(b) for a, b in zip(u, c)})
fte = pl.read_parquet(FLAT / 'test/feeds.parquet'); tts = fte['timestamp'].to_numpy().astype(np.int64); tM = fte['M'].to_numpy().astype(np.int64)
tday = (tts + TZ) // DAY; tpub = tts < PUB_TS
fva = pl.read_parquet(FLAT / 'val/feeds.parquet'); vday = (fva['timestamp'].to_numpy().astype(np.int64) + TZ) // DAY
lg('val lag (day - Sep 20):', dict(zip(*[x.tolist() for x in np.unique(vday - SEP20, return_counts=True)])))
LD = {}
for nm, sel in (('pub', tpub), ('priv', ~tpub)):
    a, b = np.unique(tday[sel] - SEP20, return_counts=True); LD[nm] = (a, b)
    lg(f'test {nm}: feeds {int(sel.sum())} (M >= 50: {int((sel & (tM >= 50)).sum())}); lag counts {dict(zip(a.tolist(), b.tolist()))}; mean (t - Sep 20 00:00 MSK) {((tts[sel] - (SEP20 * DAY - TZ)) / DAY).mean():.2f} d; '
       f'share of public by feeds {sel.mean():.3f}')
hM = fM[hold]; nH = len(hold); rows = np.concatenate([np.arange(o, o + m) for o, m in zip(fco[hold], hM)]); nR = len(rows)
fr = np.repeat(np.arange(nH), hM)
ht = fts[hold]; hd = fday[hold]; hu = fuser[hold]
a, b = np.unique(hd, return_counts=True)
lg('holdout feeds per day:', {str(datetime.date(1970, 1, 1) + datetime.timedelta(days=int(x))): int(y) for x, y in zip(a, b)})
item = np.load(FLAT / 'train/cand_item_id.npy', mmap_mode='r'); cfx = np.load(FLAT / 'train/cand_fidx.npy', mmap_mode='r')
hit = np.asarray(item[rows]).astype(np.int64)
f4 = np.asarray(np.load(FLAT / 'train/cand_f4.npy', mmap_mode='r')[rows]); ah = age_h(f4); bk = bucket(ah)
t_item = np.load(FLAT / 'test/cand_item_id.npy').astype(np.int64); t_fx = np.load(FLAT / 'test/cand_fidx.npy').astype(np.int64)
t_bk = bucket(age_h(np.load(FLAT / 'test/cand_f4.npy')))
assert (fte['fidx'].to_numpy() == np.arange(len(fte))).all()

# ------------------------------------------------------------------ 2. impressions of every needed item per train day (one chunked pass)
NI = int(max(hit.max(), t_item.max())) + 1
need = np.unique(np.concatenate([hit, t_item])); idx = np.full(NI, -1, np.int64); idx[need] = np.arange(len(need))
cnt = np.zeros(len(need) * ND, np.int64)
STEP = 16_000_000
for s in range(0, len(item), STEP):
    it = np.asarray(item[s:s + STEP]).astype(np.int64); d = fday[np.asarray(cfx[s:s + STEP])] - SEP6
    k = np.where(it < NI, idx[np.minimum(it, NI - 1)], -1); m = (k >= 0) & (d >= 0) & (d < ND)
    cnt += np.bincount(k[m] * ND + d[m], minlength=len(cnt))
    assert ((d >= 0) & (d < ND)).all()
cnt = cnt.reshape(len(need), ND)
lg(f'impressions per (item, day) counted: {len(need)} items, {int(cnt.sum())} rows of {len(item)}')
hk = idx[hit]; tk = idx[t_item]
# checks of my counter against the STORED columns (semantics of the reference and of lag 0)
st_tr = np.asarray(np.load(FC / 'train/xp_imp_c50.npy', mmap_mode='r')[rows]); st_te = np.load(FC / 'test/xp_imp_c50.npy')
cum = np.concatenate([np.zeros((len(need), 1), np.int64), np.cumsum(cnt, 1)], 1)       # cum[k, j] = impressions on days < Sep 6 + j
mine_te = np.log1p(np.minimum(cum[tk, ND], 50)).astype(np.float32)
mine_l0 = np.log1p(np.minimum(cum[hk, (hd - SEP6)[fr]], 50)).astype(np.float32)
lg(f'CHECK stored test xp_imp_c50 == log1p(min(all train impressions, 50)): max |diff| {np.abs(mine_te - st_te).max():.2e}, exact share {(mine_te == st_te).mean():.6f}')
lg(f'CHECK stored holdout xp_imp_c50 == log1p(min(impressions on days before the own day, 50)): max |diff| {np.abs(mine_l0 - st_tr).max():.2e}, exact share {(mine_l0 == st_tr).mean():.6f}')

# ------------------------------------------------------------------ 3. later rows of the same non-zero user (leave-user-out part)
hu_nz = np.unique(hu[hu != 0]); uf = np.flatnonzero(np.isin(fuser, hu_nz))
ur = np.concatenate([np.arange(o, o + m) for o, m in zip(fco[uf], fM[uf])]); uff = np.repeat(uf, fM[uf])
other = pl.DataFrame({'user': fuser[uff], 'item': np.asarray(item[ur]).astype(np.int64), 'ts2': fts[uff], 'd2': (fday[uff] - SEP6).astype(np.int64)})
lg(f'holdout users (non-zero) {len(hu_nz)}; their train feeds {len(uf)}; share of holdout feeds with user 0: {(hu == 0).mean():.4f}')


def lag_draw(nm, mode):
    a, b = LD[nm]; cdf = np.cumsum(b) / b.sum()
    if mode == 'study':   # the study's hash, re-typed from its declared recipe (uniform from a multiplicative hash of feed_id)
        un = ((ffid[hold].astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    else:
        un = np.random.default_rng(mode).random(nH)
    return a[np.minimum(np.searchsorted(cdf, un, side='right'), len(a) - 1)].astype(np.int64)


def construct(L):
    """Per row: cd = cutoff day index of the feed, D = dense cutoff day (DECLARATION: nearest day boundary to min(Sep 20, max(c_f, Sep 6 + clip(E, 0, 14 d))))."""
    cd = hd - L                                             # day index of c_f (00:00 MSK of day(t_f) - L)
    c_sec = cd * DAY - TZ
    E = c_sec[fr] - (ht[fr] - 3600.0 * ah)                  # listing age at the cutoff, seconds; NaN for capped f4
    w = np.minimum(SEP20 * DAY - TZ, np.maximum(c_sec[fr], (SEP6 * DAY - TZ) + np.clip(np.where(np.isnan(E), 1e12, E), 0, 14 * DAY)))
    D = np.floor((w + TZ) / DAY + 0.5).astype(np.int64)
    return cd, D, E


def xp_v3(L):
    """xp_imp_c50 under v3: impressions on the first (D - Sep 6) calendar days from Sep 6 that lie OUTSIDE the band |d - own day| < 3 (end capped at Sep 20),
    minus rows of the own / later feeds of the same non-zero user on those days."""
    cd, D, E = construct(L)
    dt = hd[fr] - SEP6; nk = np.clip(D - SEP6, 0, 14)       # own day index, number of exposure days to keep
    days = np.arange(14)
    combo = dt * 16 + nk; uc, inv = np.unique(combo, return_inverse=True)
    mask = np.zeros((len(uc), ND), np.int64); last = np.zeros(len(uc), np.int64)
    for i, cc in enumerate(uc):
        d0, k0 = divmod(int(cc), 16)
        ok = np.abs(days - d0) >= 3
        take = ok & (np.cumsum(ok) <= k0)
        mask[i, :14] = take; last[i] = (days[take].max() + 1) if take.any() else 0
    n = (cnt[hk] * mask[inv]).sum(1)
    # leave-user-out: rows of the same non-zero user's feeds with ts2 >= t whose day is in the window
    q = pl.DataFrame({'row': np.arange(nR), 'user': hu[fr], 'item': hit, 't': ht[fr], 'combo': inv.astype(np.int64)})
    j = q.filter(pl.col('user') != 0).join(other, on=['user', 'item'], how='inner').filter(pl.col('ts2') >= pl.col('t'))
    jc = j['combo'].to_numpy(); jd = j['d2'].to_numpy()
    inw = mask[jc, np.clip(jd, 0, ND - 1)] > 0
    sub = np.bincount(j['row'].to_numpy()[inw], minlength=nR)
    n2 = n - sub
    assert n2.min() >= 0
    fut = (mask[inv] * (np.arange(ND)[None, :] > dt[:, None])).sum(1); tot = mask[inv].sum(1)
    return np.log1p(np.minimum(n2, 50)).astype(np.float32), dict(cd=cd, D=D, E=E, win_days=tot, fut_days=fut, sub=sub)


def stats(raw, b, logt):
    x = (np.log1p(np.maximum(raw, 0)) if logt else np.clip(raw, -10, 10)).astype(np.float64); z = (np.asarray(raw) == 0).astype(np.float64)
    n = np.bincount(b, minlength=7).astype(np.float64)
    return n, np.bincount(b, weights=z, minlength=7) / np.maximum(n, 1), np.bincount(b, weights=x, minlength=7) / np.maximum(n, 1), z.mean(), x.std()


def gate(hraw, hb, traw, tb, logt, wsel=None):
    """dZ, largest bucket (>= 5% of holdout rows), dM as declared (G1-G3), holdout-row weights."""
    if wsel is not None:
        hraw, hb = hraw[wsel], hb[wsel]
    n, zh, mh, zall, _ = stats(hraw, hb, logt); nt, zt, mt, ztall, sd = stats(traw, tb, logt)
    w = n / n.sum(); gz = zh - zt
    return dict(dZ=float((w * np.abs(gz)).sum()), maxb=float(np.max(np.where(w >= 0.05, np.abs(gz), 0))), dM=float((w * np.abs(mh - mt)).sum() / sd),
                w=w, zh=zh, zt=zt, gap=gz, zero_h=zall, zero_t=ztall)


TEST = {nm: (sel[t_fx] & (tM[t_fx] >= 50)) for nm, sel in (('pub', tpub), ('priv', ~tpub))}
TESTALL = {nm: sel[t_fx] for nm, sel in (('pub', tpub), ('priv', ~tpub))}
for nm in ('priv', 'pub'):
    lg(f'reference test rows {nm}, M >= 50: {int(TEST[nm].sum())} rows of {int(((tpub if nm == "pub" else ~tpub) & (tM >= 50)).sum())} feeds; all M: {int(TESTALL[nm].sum())} rows')
OUT = {}
for nm in ('priv', 'pub'):
    print(f'\n################ view {nm} ################', flush=True)
    L = lag_draw(nm, 'study'); Ls = np.load(STUDY / f'view_{nm}_L.npy'); Ds = np.load(STUDY / f'view_{nm}_D.npy')
    xp, info = xp_v3(L)
    lg(f'lags imposed (study hash, my re-typing): {dict(zip(*[x.tolist() for x in np.unique(L, return_counts=True)]))}; equal to the study\'s view_{nm}_L: {bool(np.array_equal(L, Ls))}; '
       f'my D equal to the study\'s view_{nm}_D on {np.mean(info["D"] == Ds):.6f} of rows')
    xs = np.load(STUDY / f'lag_{nm}_v3_xp_imp_c50.npy')
    lg(f'MY REBUILD of xp_imp_c50 ({nm}_v3) vs the study\'s cached column: exact on {np.mean(xp == xs):.6f} of {nR} rows, max |diff| {np.abs(xp - xs).max():.3g}; zero share mine {np.mean(xp == 0):.4f} study {np.mean(xs == 0):.4f}')
    np.save(MYC / f'my_{nm}_v3_xp_imp_c50.npy', xp)
    D = info['D']; dtr = hd[fr]
    full = D >= SEP20; plain = D == info['cd'][fr]; emp = D <= SEP6
    lg(f'what the "lag" is, by row: dense cutoff D = Sep 20 (full dense window, only the +-2-day band removed) {full.mean():.3f} of rows; D = c_f (plain counts up to T - lag) {plain.mean():.3f} '
       f'(of which empty dense window, D <= Sep 6: {emp.mean():.3f} of all rows); in between (window shifted in the listing\'s life) {(~full & ~plain).mean():.3f}')
    lg(f'  rows with D later than the own day (dense counts may include days AFTER the row): {np.mean(D > dtr + 1):.3f}; '
       f'xp_imp_c50 window: mean share of window days that are after the own day, rows with a non-empty window: {np.mean(info["fut_days"][info["win_days"] > 0] / info["win_days"][info["win_days"] > 0]):.3f}')
    for b in range(7):
        m = bk == b
        print(f'      {BN[b]:9s} rows {m.mean():.3f} | D = Sep 20: {full[m].mean():.3f} | D = c_f: {plain[m].mean():.3f} | D > own day: {np.mean(D[m] > dtr[m] + 1):.3f}')
    cols = {'it_hist_all_luo': (np.load(STUDY / f'lag_{nm}_v3_it_hist_all_luo.npy'), True), 'seq_cv_pop': (np.load(STUDY / f'lag_{nm}_v3_seq_cv_pop.npy'), True),
            'xp_imp_c50 [my rebuild]': (xp, False), 'xp_imp_c50 [study file]': (xs, False)}
    res = {}
    for cn, (hraw, logt) in cols.items():
        base = cn.split(' ')[0]
        l0 = np.asarray(np.load(FC / 'train' / f'{base}.npy', mmap_mode='r')[rows]); tr = np.load(FC / 'test' / f'{base}.npy')
        g0 = gate(l0, bk, tr[TEST[nm]], t_bk[TEST[nm]], logt); g1 = gate(hraw, bk, tr[TEST[nm]], t_bk[TEST[nm]], logt)
        ga = gate(hraw, bk, tr[TESTALL[nm]], t_bk[TESTALL[nm]], logt)
        late = (hd >= SEP6 + 7)[fr]; gl = gate(hraw, bk, tr[TEST[nm]], t_bk[TEST[nm]], logt, late)
        res[cn] = dict(dZ0=g0['dZ'], dZ=g1['dZ'], maxb=g1['maxb'], dM=g1['dM'])
        print(f'\n {cn}: dZ lag0 {g0["dZ"]:.4f} -> lagged {g1["dZ"]:.4f} | largest bucket (>= 5% rows) {g1["maxb"]:.4f} | dM {g1["dM"]:.4f} | pooled zero share lag0 {g0["zero_h"]:.3f} lagged {g1["zero_h"]:.3f} test {g1["zero_t"]:.3f}')
        print(f'   reference = ALL-M test rows instead of M >= 50: dZ {ga["dZ"]:.4f} largest bucket {ga["maxb"]:.4f} dM {ga["dM"]:.4f}   |   holdout days >= Sep 13 only ({int(late.sum())} rows): dZ {gl["dZ"]:.4f} largest bucket {gl["maxb"]:.4f} dM {gl["dM"]:.4f}')
        print('   bucket     rows   zero lag0 / lagged / test     gap(lagged - test)')
        for b in range(7):
            print(f'   {BN[b]:9s} {g1["w"][b]:.3f}   {g0["zh"][b]:.3f} / {g1["zh"][b]:.3f} / {g1["zt"][b]:.3f}     {g1["gap"][b]:+.4f}' + ('   <-- > 0.10 but bucket < 5% of rows: not counted by G2' if abs(g1['gap'][b]) > 0.10 and g1['w'][b] < 0.05 else ''))
        fails = []
        if g1['dZ'] > 0.05: fails.append('G1')
        if g1['maxb'] > 0.10: fails.append('G2')
        if g1['dM'] > 0.20: fails.append('G3')
        if base == 'it_hist_all_luo' and g1['dZ'] > g0['dZ'] / 3: fails.append('G4')
        print(f'   criteria G1 (dZ <= 0.05), G2 (bucket <= 0.10), G3 (dM <= 0.20)' + (', G4 (dZ <= lag-0 dZ / 3)' if base == 'it_hist_all_luo' else '') + f': {"PASS" if not fails else "FAIL " + ",".join(fails)}')
    # residual of the failing bucket by holdout day group
    tr = np.load(FC / 'test/xp_imp_c50.npy'); zt = float((tr[TEST[nm] & (t_bk == 4)] == 0).mean())
    for lo, hi, lab in ((0, 3, 'Sep 6-8'), (3, 7, 'Sep 9-12'), (7, 9, 'Sep 13-14'), (11, 14, 'Sep 17-19')):
        m = (bk == 4) & (dtr - SEP6 >= lo) & (dtr - SEP6 < hi)
        print(f'   xp_imp_c50 10-24d bucket, holdout days {lab}: zero share {np.mean(xp[m] == 0):.3f} vs test {zt:.3f}: gap {np.mean(xp[m] == 0) - zt:+.3f} ({int(((hd - SEP6 >= lo) & (hd - SEP6 < hi)).sum())} feeds)')
    # sensitivity of the failing criterion to the lag draw (the study did not test it)
    sens = []
    for seed in (1, 2, 3, 4, 5):
        x2, _ = xp_v3(lag_draw(nm, seed)); g2 = gate(x2, bk, tr[TEST[nm]], t_bk[TEST[nm]], False); sens.append((g2['dZ'], g2['maxb'], g2['gap'][4], g2['gap'][3]))
    print(f'   xp_imp_c50 under 5 other random lag draws: dZ {min(s[0] for s in sens):.4f}..{max(s[0] for s in sens):.4f}; largest bucket {min(s[1] for s in sens):.4f}..{max(s[1] for s in sens):.4f}; '
          f'10-24d gap {min(s[2] for s in sens):+.4f}..{max(s[2] for s in sens):+.4f}; 7-10d gap {min(s[3] for s in sens):+.4f}..{max(s[3] for s in sens):+.4f}', flush=True)
    OUT[nm] = res
json.dump(OUT, open(MYC / 'v_gate.json', 'w'), indent=1)
lg('done')
